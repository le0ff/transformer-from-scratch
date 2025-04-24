from typing import Any

import numpy as np
from numpy import ndarray

from src.layers.base import BaseLayer


class ReLU(BaseLayer):
    """
    Rectified Linear Unit activation layer.

    Applies the ReLU activation function f(x) = max(0, x) element-wise to the input.
    """

    def __init__(self) -> None:
        """
        Initializes ReLU layer.
        """
        super().__init__()
        self._input_cache = None

    def forward(self, x: ndarray, **kwargs: Any) -> ndarray:
        """
        Forward pass through the ReLU layer.

        Parameters:
            x (ndarray): Input data.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ndarray: Output data after applying ReLU activation.
        """
        # store input for backward pass
        self._input_cache = x
        return np.maximum(0, x)

    # def backward(self, grad_output: ndarray) -> ndarray:
    #     """
    #     Backward pass through the ReLU layer.

    #     Parameters:
    #         grad_output (ndarray): Gradient of the loss with respect to the output.

    #     Returns:
    #         ndarray: Gradient of the loss with respect to the input.
    #     """
    #     if self._input_cache is None:
    #         raise ValueError("No input cache found. Forward pass must be called first.")

    #     relu_grad = (self._input_cache > 0).astype(grad_output.dtype)
    #     return grad_output * relu_grad
