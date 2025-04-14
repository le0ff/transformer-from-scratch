from typing import Any, Optional

import numpy as np
from numpy import ndarray

from src.layers.base import BaseLayer


class Dropout(BaseLayer):
    """
    Dropout layer for regularization.

    During training, randomly zeroes some elements of the input tensor with probability 'rate' and scales the remaining elements by 1 / (1 - rate).
    During evaluation, it returns the input tensor unchanged.
    """

    def __init__(self, rate: float, seed: Optional[int] = None) -> None:
        """
        Initializes the Dropout layer.

        Parameters:
            rate (float): The dropout rate, i.e., the probability of setting
                          an element to zero. Must be in [0.0, 1.0).
            seed (Optional[int]): Seed for the random number generator used
                                  for creating dropout masks. Ensures reproducibility
                                  during training if set.
        """
        super().__init__()

        if not isinstance(rate, (int, float)) or not (0.0 <= rate < 1.0):
            raise ValueError("Dropout rate must be a float in [0.0, 1.0).")

        self.rate = float(rate)
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            if not isinstance(seed, int):
                raise ValueError("Seed must be an integer.")
            self.rng = np.random.default_rng(seed)

        # Cache for mask used during training
        self._mask = None

    def forward(self, x: ndarray, **kwargs: Any) -> ndarray:
        """
        Forward pass through the Dropout layer.

        Parameters:
            x (ndarray): Input data.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ndarray: Output data after applying dropout.
        """
        if self.training:
            # Create mask with same shape as x, where each element is 0.0 with probability 'rate'
            self._mask = (self.rng.random(x.shape) >= self.rate).astype(x.dtype)

            scaling_factor = 1.0 / (1.0 - self.rate)
            # Apply mask and scale the output
            return x * self._mask * scaling_factor
        else:
            self._mask = None
            return x

    # def backward(self, grad_output: ndarray) -> ndarray:
    #     """
    #     Backward pass through the Dropout layer.

    #     Parameters:
    #         grad_output (ndarray): Gradient of the loss with respect to the output.

    #     Returns:
    #         ndarray: Gradient of the loss with respect to the input.
    #     """
    #     if not self.training:
    #         return grad_output

    #     if self._mask is None:
    #         raise ValueError("No mask found. Forward pass must be called first.")

    #     scaling_factor = 1.0 / (1.0 - self.rate)
    #     grad_input = grad_output * self._mask * scaling_factor
    #     return grad_input
