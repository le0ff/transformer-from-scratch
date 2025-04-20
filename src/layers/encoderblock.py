from typing import Dict, Optional

import numpy as np
from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.feedforward import FeedForwardBlock
from src.layers.multiheadattentionblock import MultiHeadAttentionBlock
from src.layers.residual import ResidualConnection


class EncoderBlock(BaseLayer):
    """
    Encoder Block for the Transformer model.

    This block consists of a multi-head self-attention layer and a feed-forward neural network.
    It also includes dropout and residual connections.
    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
        seed: Optional[int],
    ) -> None:
        """
        Initializes the EncoderBlock.

        Parameters:
            self_attention_block (MultiHeadAttentionBlock): The multi-head self-attention block.
            feed_forward_block (FeedForwardBlock): The feed-forward block.
            dropout (float): Dropout rate.
            seed (Optional[int]): Seed for random number generator.
        """
        super().__init__()

        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout < 1.0):
            raise ValueError("Dropout must be a float in [0.0, 1.0).")

        if seed is not None and not isinstance(seed, int):
            raise ValueError("Seed must be an integer.")

        if seed is not None:
            main_rng = np.random.default_rng(seed)
            max_seed_val = 2**31 - 1
            seeds = main_rng.integers(0, max_seed_val, size=2)
            seed_residual1 = int(seeds[0])
            seed_residual2 = int(seeds[1])
        else:
            seed_residual1 = None
            seed_residual2 = None

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # 2 reisudal connections with dropout
        d_model = self.self_attention_block.d_model
        eps = 1e-5
        self.residual1 = ResidualConnection(
            normalized_shape=d_model, eps=eps, dropout_rate=dropout, seed=seed_residual1
        )
        self.residual2 = ResidualConnection(
            normalized_shape=d_model, eps=eps, dropout_rate=dropout, seed=seed_residual2
        )

        self._layers = {
            "self_attention": self.self_attention_block,
            "feed_forward": self.feed_forward_block,
            "residual1": self.residual1,
            "residual2": self.residual2,
        }

    def forward(self, x: ndarray, mask: ndarray) -> ndarray:
        """
        Forward pass through the encoder block.

        """
        x = self.residual1(
            x, sublayer=lambda x: self.self_attention_block(x, x, x, mask)
        )
        x = self.residual2(x, sublayer=self.feed_forward_block)
        return x

    def train(self) -> None:
        """Set block to training mode."""
        super().train()
        for layer in self._layers.values():
            layer.train()

    def eval(self) -> None:
        """Set block to evaluation mode."""
        super().eval()
        for layer in self._layers.values():
            layer.eval()

    def get_parameters(self) -> Dict[str, ndarray]:
        """Get all parameters from sublayers, with unique prefixes."""
        params = {}
        for name, layer in self._layers.items():
            sub_params = layer.get_parameters()
            for key, value in sub_params.items():
                params[f"{name}_{key}"] = value
        return params

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """Set parameters for all sublayers, expecting unique prefixes."""
        # Prepare a dict for each sublayer
        sublayer_param_dicts = {name: {} for name in self._layers}
        processed_keys = set()

        # Distribute parameters to the correct sublayer dict
        for key, value in params.items():
            matched = False
            for sublayer in self._layers:
                prefix = f"{sublayer}_"
                if key.startswith(prefix):
                    sublayer_param_dicts[sublayer][key[len(prefix) :]] = value
                    processed_keys.add(key)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unexpected parameter key for EncoderBlock: {key}")

        # Set parameters for each sublayer
        for sublayer, sub_params in sublayer_param_dicts.items():
            if sub_params:  # Only set if there are parameters
                self._layers[sublayer].set_parameters(sub_params)

        # Check for missing keys
        if len(processed_keys) != len(params):
            missing = set(params.keys()) - processed_keys
            raise ValueError(f"Some parameters were not processed: {missing}")
