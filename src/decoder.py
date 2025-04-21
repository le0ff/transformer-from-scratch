from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.decoderblock import DecoderBlock
from src.layers.feedforward import FeedForwardBlock
from src.layers.multiheadattentionblock import MultiHeadAttentionBlock
from src.layers.normalization import LayerNorm


class Decoder(BaseLayer):
    """
    Transformer Decoder consisting of a sequence of Decoder blocks.
    """

    def __init__(self, layers: List[BaseLayer]) -> None:
        """
        Initializes the Decoder.

        Parameters:
            layers (List[BaseLayer]): List of Decoder blocks in order.
        """
        super().__init__()
        self.layers = layers
        # Use d_model from first decoder block to initialize final normalization
        first_block = layers[0]
        self.norm = LayerNorm(first_block.self_attention_block.d_model)

    @classmethod
    def from_config(
        cls,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_blocks: int,
        seed: Optional[int] = None,
    ) -> "Decoder":
        """
        Alternate constructor to build a Decoder from config.

        Parameters:
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            d_ff (int): Feedforward dimension.
            dropout (float): Dropout rate.
            num_blocks (int): Number of decoder blocks.
            seed (Optional[int]): Seed for reproducibility.

        Returns:
            Decoder: An initialized Decoder instance.
        """
        main_rng = np.random.default_rng(seed)
        max_seed_val = 2**31 - 1
        layers = []

        for _ in range(num_blocks):
            seeds = main_rng.integers(0, max_seed_val, size=4)
            self_attn_seed = int(seeds[0])
            cross_attn_seed = int(seeds[1])
            ff_seed = int(seeds[2])
            block_seed = int(seeds[3])

            self_attn = MultiHeadAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout_rate=dropout,
                seed=self_attn_seed,
            )
            cross_attn = MultiHeadAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout_rate=dropout,
                seed=cross_attn_seed,
            )
            ff = FeedForwardBlock(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                seed=ff_seed,
            )

            block = DecoderBlock(
                self_attention_block=self_attn,
                cross_attention_block=cross_attn,
                feed_forward_block=ff,
                dropout=dropout,
                seed=block_seed,
            )
            layers.append(block)

        return cls(layers=layers)

    def forward(
        self,
        x: ndarray,
        encoder_output: ndarray,
        src_mask: ndarray,
        tgt_mask: ndarray,
    ) -> ndarray:
        """
        Forward pass through the Decoder.

        Parameters:
            x (ndarray): Input to the decoder (target embeddings).
            encoder_output (ndarray): Output from the encoder.
            src_mask (ndarray): Padding mask for encoder output.
            tgt_mask (ndarray): Causal mask for decoder self-attention.

        Returns:
            ndarray: Output from the decoder.
        """
        for layer in self.layers:
            x = layer(
                x, encoder_output=encoder_output, tgt_mask=tgt_mask, src_mask=src_mask
            )
        return self.norm(x)

    def train(self) -> None:
        """Set decoder and submodules to training mode."""
        super().train()
        for layer in self.layers:
            layer.train()
        self.norm.train()

    def eval(self) -> None:
        """Set decoder and submodules to evaluation mode."""
        super().eval()
        for layer in self.layers:
            layer.eval()
        self.norm.eval()

    def get_parameters(self) -> Dict[str, ndarray]:
        """Get all parameters from the decoder, namespaced by layer and sublayer."""
        params = {}
        for idx, layer in enumerate(self.layers):
            layer_params = layer.get_parameters()
            for key, value in layer_params.items():
                params[f"layer{idx}_{key}"] = value
        norm_params = self.norm.get_parameters()
        for key, value in norm_params.items():
            params[f"norm_{key}"] = value
        return params

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """Set decoder parameters from a flat dict with namespaced keys."""
        if not params:
            raise ValueError("No parameters provided for Decoder.")

        processed_keys = set()
        layer_param_dicts = [{} for _ in self.layers]
        norm_param_dict = {}

        for key, value in params.items():
            if key.startswith("layer"):
                parts = key.split("_", 1)
                if len(parts) != 2 or not parts[0][5:].isdigit():
                    raise ValueError(f"Invalid layer key format: {key}")
                idx = int(parts[0][5:])
                if idx >= len(self.layers):
                    raise ValueError(f"Layer index out of range: {idx}")
                layer_param_dicts[idx][parts[1]] = value
                processed_keys.add(key)
            elif key.startswith("norm_"):
                norm_param_dict[key[len("norm_") :]] = value
                processed_keys.add(key)
            else:
                raise ValueError(f"Unexpected parameter key for Decoder: {key}")

        for idx, layer_params in enumerate(layer_param_dicts):
            if layer_params:
                self.layers[idx].set_parameters(layer_params)

        if norm_param_dict:
            self.norm.set_parameters(norm_param_dict)

        if processed_keys != set(params):
            missing = set(params) - processed_keys
            raise ValueError(f"Some parameters were not processed: {missing}")
