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
        self.gamma = np.ones((normalized_shape,), dtype=np.float32)
        self.beta = np.zeros((normalized_shape,), dtype=np.float32)

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
        # Introduced for backprop later
        std = np.sqrt(variance + self.eps)
        x_hat = (x - mean) / np.sqrt(variance + self.eps)
        x_norm = x_hat * self.gamma + self.beta

        self._x = x
        self._x_norm = x_norm
        self._mean = mean
        self._std = std

        return x_norm

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get the parameters of the layer.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing gamma and beta if affine is True.
        """
        params = {}
        params["gamma"] = self.gamma
        params["beta"] = self.beta
        return params

    # def backward(self, dout: np.ndarray) -> np.ndarray:
    #     """
    #     Backward pass for LayerNorm.

    #     Parameters:
    #         dout (np.ndarray): Gradient of the loss w.r.t. the output.

    #     Returns:
    #         np.ndarray: Gradient of the loss w.r.t. the input.
    #     """
    #     x = self._x
    #     x_norm = self._x_norm
    #     std = self._std

    #     N = x.shape[-1]  # normalized shape (e.g., feature dim)

    #     # Initialize gradients
    #     dx_norm = dout * self.gamma if self.affine else dout

    #     # Backprop through normalization
    #     dx = (
    #         (1.0 / N)
    #         * (1.0 / std)
    #         * (
    #             N * dx_norm
    #             - np.sum(dx_norm, axis=-1, keepdims=True)
    #             - x_norm * np.sum(dx_norm * x_norm, axis=-1, keepdims=True)
    #         )
    #     )

    #     # Gradients for gamma and beta
    #     if self.affine:
    #         self._dgamma = np.sum(dout * x_norm, axis=0)
    #         self._dbeta = np.sum(dout, axis=0)

    #     return dx
