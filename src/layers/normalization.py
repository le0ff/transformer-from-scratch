from typing import Any, Dict

import numpy as np

from src.layers.base import BaseLayer


class LayerNorm(BaseLayer):
    """
    Layer Normalization Layer as introduced in Layer Normalization" (Ba et al., 2016).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, affine: bool = True):
        """
        Initialize the LayerNorm layer.

        Parameters:
            normalized_shape (int): The number of features in the input to normalize.
            eps (float): A small value to prevent division by zero.
            affine (bool): Whether to use learnable affine parameters (gamma and beta).
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = np.ones((normalized_shape,), dtype=np.float32)
            self.beta = np.zeros((normalized_shape,), dtype=np.float32)
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply layer normalization to the input.

        Parameters:
            x (np.ndarray): Input data of shape (..., normalized_shape).

        Returns:
            np.ndarray: Normalized and optionally affine-transformed output.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get the parameters of the layer.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing gamma and beta if affine is True.
        """
        params = {}
        if self.affine:
            params["gamma"] = self.gamma
            params["beta"] = self.beta
        return params
