from typing import Any, Dict

import numpy as np

from src.layers.base import BaseLayer


class LayerNorm(BaseLayer):
    """
    Layer Normalization Layer as introduced in Layer Normalization" (Ba et al., 2016).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Initialize the LayerNorm layer.

        Parameters:
            normalized_shape (int): The number of features in the input to normalize.
            eps (float): A small value to prevent division by zero.
        """
        super().__init__()

        if not isinstance(normalized_shape, int) or normalized_shape <= 0:
            raise ValueError("`normalized_shape` must be a positive integer.")

        if not isinstance(eps, (float, np.floating)) or eps <= 0.0:
            raise ValueError("`eps` must be a positive float.")

        self.normalized_shape = normalized_shape
        self.eps = eps
        # default values for adaptive gain and bias as described in the paper
        self.gamma = np.ones((normalized_shape,), dtype=np.float32)
        self.beta = np.zeros((normalized_shape,), dtype=np.float32)

    def forward(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply layer normalization to the input.

        Parameters:
            x (np.ndarray): Input data of shape (..., normalized_shape).

        Returns:
            np.ndarray: Normalized output.
        """
        # Check if last axis dim is equal to normalized_shape
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"Expected input's last dimension to be {self.normalized_shape}, "
                f"but got {x.shape[-1]}."
            )

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
            Dict[str, np.ndarray]: Dictionary containing gamma and beta.
        """
        params = {"gamma": self.gamma, "beta": self.beta}

        return params

    def set_parameters(self, params: Dict[str, np.ndarray]):
        """
        Set the parameters for the normalization layer

        Parameters:
            params(Dict[str, ndarray]): Dictionary of parameters
        """
        # Set gain
        if "gamma" in params:
            if params["gamma"].shape != (self.normalized_shape,):
                raise ValueError(
                    f"Expected gamma shape {(self.normalized_shape,)}, "
                    f"but got {params['gamma'].shape}"
                )
            self.gamma = params["gamma"]
        else:
            raise ValueError("Scale parameter 'gamma' missing in params dictionary.")

        # Set bias
        if "beta" in params:
            if params["beta"].shape != (self.normalized_shape,):
                raise ValueError(
                    f"Expected beta shape {(self.normalized_shape,)}, "
                    f"but got {params['beta'].shape}"
                )
            self.beta = params["beta"]
        else:
            raise ValueError("Bias parameter 'beta' missing in params dictionary.")

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
    #     dx_norm = dout * self.gamma

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
    #
    #      self._dgamma = np.sum(dout * x_norm, axis=0)
    #      self._dbeta = np.sum(dout, axis=0)

    #     return dx
