from typing import Optional

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

    def forward(self, x: ndarray, mask: ndarray) -> ndarray:
        """
        Forward pass through the encoder block.

        """
        x = self.residual1(x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual2(x, self.feed_forward_block)
        return x

    def train(self) -> None:
        """Set block to training mode."""
        super().train()
        self.self_attention_block.train()
        self.feed_forward_block.train()
        self.residual1.train()
        self.residual2.train()

    def eval(self) -> None:
        """Set block to evaluation mode."""
        super().eval()
        self.self_attention_block.eval()
        self.feed_forward_block.eval()
        self.residual1.eval()
        self.residual2.eval()
