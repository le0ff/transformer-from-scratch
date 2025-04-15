from typing import Any, Optional

import numpy as np
from numpy import ndarray

from src.layers.activations.relu import ReLU
from src.layers.base import BaseLayer
from src.layers.dropout import Dropout
from src.layers.linear import Linear


class FeedForwardBlock(BaseLayer):
    """
    Feedforward block applying x -> Linear -> ReLU -> Dropout -> Linear.
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float, seed: Optional[int] = None
    ) -> None:
        """
        Initializes the FeedForwardBlock.

        Parameters:
            d_model (int): Dimension of the input and output.
            d_ff (int): Dimension of the feedforward layer.
            dropout (float): Dropout rate.
            seed (Optional[int]): Seed for random number generator.
        """
        super().__init__()

        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive integers.")

        if seed is not None and not isinstance(seed, int):
            raise ValueError("Seed must be an integer.")

        if seed is not None:
            main_rng = np.random.default_rng(seed)
            max_seed_val = 2**31 - 1
            seeds = main_rng.integers(0, max_seed_val, size=3)
            linear1_seed = int(seeds[0])
            dropout_seed = int(seeds[1])
            linear2_seed = int(seeds[2])
        else:
            linear1_seed = None
            dropout_seed = None
            linear2_seed = None

        self.linear1 = Linear(d_model, d_ff, use_bias=True, seed=linear1_seed)
        self.relu = ReLU()
        self.dropout = Dropout(dropout, seed=dropout_seed)
        self.linear2 = Linear(d_ff, d_model, use_bias=True, seed=linear2_seed)

        self._layers = [
            self.linear1,
            self.relu,
            self.dropout,
            self.linear2,
        ]

        def train(self) -> None:
            """Set the layer to training mode."""
            self.training = True
            for layer in self._layers:
                layer.train()

        def eval(self) -> None:
            """Set the layer to evaluation mode."""
            self.training = False
            for layer in self._layers:
                layer.eval()

        def forward(self, x: ndarray, **kwargs: Any) -> ndarray:
            """
            Forward pass through the feedforward block.

            Parameters:
                x (ndarray): Input tensor of shape (batch_size, seq_len, d_model).

            Returns:
                ndarray: Output tensor of shape (batch_size, seq_len, d_model).
            """
            for layer in self._layers:
                x = layer.forward(x)
            return x
