from typing import Dict, Optional

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

    def __init__(self, layers: Dict[str, BaseLayer]) -> None:
        """
        Initializes the Decoder.

        Parameters:
            layers (Dict[str, BaseLayer]): Dictionary of layers in the decoder (in sequential order).
        """
        super().__init__()
        self.layers = layers
        # Use d_model from first decoder block to initialize final normalization
        first_block = next(iter(layers.values()))
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
        layers = {}

        for i in range(num_blocks):
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

            layers[f"block{i}"] = DecoderBlock(
                self_attention_block=self_attn,
                cross_attention_block=cross_attn,
                feed_forward_block=ff,
                dropout=dropout,
                seed=block_seed,
            )

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
        for layer in self.layers.values():
            x = layer(
                x, encoder_output=encoder_output, tgt_mask=tgt_mask, src_mask=src_mask
            )
        return self.norm(x)

    def train(self) -> None:
        """Set decoder and submodules to training mode."""
        super().train()
        for layer in self.layers.values():
            layer.train()
        self.norm.train()

    def eval(self) -> None:
        """Set decoder and submodules to evaluation mode."""
        super().eval()
        for layer in self.layers.values():
            layer.eval()
        self.norm.eval()

    def get_parameters(self) -> Dict[str, ndarray]:
        """Get all parameters from the decoder, namespaced by layer and sublayer."""
        params = {}
        for block_name, block in self.layers.items():
            block_params = block.get_parameters()
            for key, value in block_params.items():
                params[f"{block_name}_{key}"] = value
        # Add LayerNorm parameters for the encoder output norm
        norm_params = self.norm.get_parameters()
        for key, value in norm_params.items():
            params[f"norm_{key}"] = value
        return params

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """Set decoder parameters from a flat dict with namespaced keys."""
        if not params:
            raise ValueError("No parameters provided for Decoder.")

        # Prepare dicts for each block and for norm
        block_param_dicts = {name: {} for name in self.layers}
        norm_param_dict = {}
        processed_keys = set()

        for key, value in params.items():
            matched = False
            # Check if key belongs to a block
            for block_name in self.layers:
                prefix = f"{block_name}_"
                if key.startswith(prefix):
                    block_param_dicts[block_name][key[len(prefix) :]] = value
                    processed_keys.add(key)
                    matched = True
                    break
            # Check if key belongs to norm
            if not matched and key.startswith("norm_"):
                norm_param_dict[key[len("norm_") :]] = value
                processed_keys.add(key)
                matched = True
            if not matched:
                raise ValueError(f"Unexpected parameter key for Decoder: {key}")

        # Set parameters for each block
        for block_name, block_params in block_param_dicts.items():
            if block_params:
                self.layers[block_name].set_parameters(block_params)
        # Set parameters for norm
        if norm_param_dict:
            self.norm.set_parameters(norm_param_dict)

        # Check for missing keys
        if len(processed_keys) != len(params):
            missing = set(params.keys()) - processed_keys
            raise ValueError(f"Some parameters were not processed: {missing}")
