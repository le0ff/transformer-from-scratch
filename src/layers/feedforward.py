from typing import Any, Optional

import numpy as np
from numpy import ndarray

from src.layers.activations.relu import ReLU
from src.layers.base import BaseLayer
from src.layers.dropout import Dropout
from src.layers.linear import Linear


class FeedForwardBlock(BaseLayer):
    """
    Implements Feed Forward Block typically found in Transformer architectures.

    Consists of two linear layers with a ReLU activation and dropout in between.
    x -> Linear1 -> ReLU -> Dropout -> Linear2
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

        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout < 1.0):
            raise ValueError("Dropout must be a float in [0.0, 1.0).")

        if seed is not None and not isinstance(seed, int):
            raise ValueError("Seed must be an integer.")

        # store configuration
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout

        # generate random seeds for layers if seed is provided
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

        # Initialize layers
        self.linear1 = Linear(d_model, d_ff, use_bias=True, seed=linear1_seed)
        self.relu = ReLU()
        self.dropout_layer = Dropout(dropout, seed=dropout_seed)
        self.linear2 = Linear(d_ff, d_model, use_bias=True, seed=linear2_seed)

        # dictionary of layers with meaningful names for parameter handling & correct order for forward/backward pass
        self._layers = {
            "linear1": self.linear1,
            "relu": self.relu,
            "dropout": self.dropout_layer,
            "linear2": self.linear2,
        }

    def train(self) -> None:
        """Set the layer to training mode."""
        super().train()
        for layer in self._layers:
            layer.train()

    def eval(self) -> None:
        """Set the layer to evaluation mode."""
        super().eval()
        for layer in self._layers:
            layer.eval()

    def forward(self, x: ndarray, **kwargs: Any) -> ndarray:
        """
        Forward pass through the feedforward block by iterating through layers dictionary in order.

        Parameters:
            x (ndarray): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            ndarray: Output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self._layers.values():
            x = layer.forward(x)
        return x
