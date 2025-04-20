from typing import Dict, Optional

import numpy as np
from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.feedforward import FeedForwardBlock
from src.layers.multiheadattentionblock import MultiHeadAttentionBlock
from src.layers.residual import ResidualConnection


class DecoderBlock(BaseLayer):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
        seed: Optional[int],
    ) -> None:
        """
        Initializes the DecoderBlock.

        Parameters:
            self_attention_block (MultiHeadAttentionBlock): The multi-head self-attention block.
            cross_attention_block  : multi‑head block for encoder–decoder attention
            feed_forward_block (FeedForwardBlock): The feed-forward block.
            dropout (float): Dropout rate.
            seed (Optional[int]): Seed for random number generator.
        """
        super().__init__()

        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an int or None.")

        if seed is not None:
            main_rng = np.random.default_rng(seed)
            max_seed_val = 2**31 - 1
            seeds = main_rng.integers(0, max_seed_val, size=3)
            seed_residual1 = int(seeds[0])
            seed_residual2 = int(seeds[1])
            seed_residual3 = int(seeds[2])
        else:
            seed_residual1 = None
            seed_residual2 = None
            seed_residual3 = None

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        d_model = self_attention_block.d_model
        eps = 1e-5

        self.residual1 = ResidualConnection(
            normalized_shape=d_model, eps=eps, dropout_rate=dropout, seed=seed_residual1
        )
        self.residual2 = ResidualConnection(
            normalized_shape=d_model, eps=eps, dropout_rate=dropout, seed=seed_residual2
        )
        self.residual3 = ResidualConnection(
            normalized_shape=d_model, eps=eps, dropout_rate=dropout, seed=seed_residual3
        )

    def forward(
        self, x: ndarray, encoder_output: ndarray, tgt_mask: ndarray, src_mask: ndarray
    ) -> ndarray:
        """
        Args
        ----
        x              : decoder input  (batch, tgt_len, d_model)
        encoder_output : encoder memory (batch, src_len, d_model)
        tgt_mask       : causal mask for masked self‑attention
                         (batch, 1, tgt_len, tgt_len)   or broadcast‑compatible
        src_mask       : padding mask for cross‑attention
                         (batch, 1, 1, src_len)         or broadcast‑compatible

        Returns
        -------
        ndarray (batch, tgt_len, d_model)
        """
        x = self.residual1(
            x, sublayer=lambda x_: self.self_attention_block(x_, x_, x_, tgt_mask)
        )

        x = self.residual2(
            x,
            sublayer=lambda x_: self.cross_attention_block(
                x_, encoder_output, encoder_output, src_mask
            ),
        )

        x = self.residual3(x, sublayer=self.feed_forward_block)

        # return x.astype(np.float32)
        return x

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Return *one flat dict* whose keys are prefixed with the sub‑layer name followed by “_”.
        Sub‑layers that expose no parameters are silently skipped.
        """
        param_dict: Dict[str, np.ndarray] = {}

        # name → attribute mapping
        sublayers = {
            "self_attn": self.self_attention_block,
            "cross_attn": self.cross_attention_block,
            "ff": self.feed_forward_block,
            "residual1": self.residual1,
            "residual2": self.residual2,
            "residual3": self.residual3,
        }

        for prefix, layer in sublayers.items():
            sub_p = layer.get_parameters()
            for k, v in sub_p.items():
                param_dict[f"{prefix}_{k}"] = v

        return param_dict

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """
        Load parameters from a *flat* dict created by `get_parameters`.

        Each key must start with a recognised prefix; the remainder of the
        key is forwarded to the corresponding sub‑layer’s `set_parameters`.
        """
        # Build empty mail‑boxes for every sub‑layer that owns params
        sublayers = {
            "self_attn": self.self_attention_block,
            "cross_attn": self.cross_attention_block,
            "ff": self.feed_forward_block,
            "residual1": self.residual1,
            "residual2": self.residual2,
            "residual3": self.residual3,
        }
        mailboxes = {
            name: {} for name, layer in sublayers.items() if layer.get_parameters()
        }

        processed: set[str] = set()

        # Route keys by prefix
        for full_key, tensor in params.items():
            for prefix in mailboxes:
                tag = prefix + "_"
                if full_key.startswith(tag):
                    short = full_key[len(tag) :]
                    mailboxes[prefix][short] = tensor
                    processed.add(full_key)
                    break
            else:
                # no matching prefix
                raise ValueError(f"Unrecognised parameter key: {full_key}")

        # Dispatch into each sub‑layer
        for prefix, sub_params in mailboxes.items():
            if not sub_params:
                continue
            try:
                sublayers[prefix].set_parameters(sub_params)
            except ValueError as e:
                raise ValueError(
                    f"Error setting parameters for sub-layer '{prefix}': {e}"
                ) from e

        # 3) Catch missing keys
        if processed != set(params):
            missing = set(params) - processed
            raise ValueError(f"Missing parameters for layers: {missing}")

    def train(self) -> None:
        """Put block and all sub‑modules into training mode."""
        super().train()
        self.self_attention_block.train()
        self.cross_attention_block.train()
        self.feed_forward_block.train()
        self.residual1.train()
        self.residual2.train()
        self.residual3.train()

    def eval(self) -> None:
        """Put block and all sub‑modules into eval mode."""
        super().eval()
        self.self_attention_block.eval()
        self.cross_attention_block.eval()
        self.feed_forward_block.eval()
        self.residual1.eval()
        self.residual2.eval()
        self.residual3.eval()
