from typing import Any, Dict

import numpy as np

from src.layers.base import BaseLayer


class Linear(BaseLayer):
    """
    Standard fully connected linear layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initializes Linear layer.

        Parameters:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # He initialization for weights
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / self.input_dim)
        self.b = np.zeros(output_dim)

        # Cache for backward pass
        self._input_cache = None

    def forward(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Forward pass through the layer.

        Parameters:
            x (np.ndarray): Input data.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            np.ndarray: Output data.
        """
        # store input for backward pass
        self._input_cache = x

        return x @ self.W + self.b

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get the parameters of the layer.

        Returns:
            Dict[str, np.ndarray]: Dictionary of parameters.
        """
        return {"W": self.W, "b": self.b}

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """
        Set the parameters of the layer.

        Parameters:
            params (Dict[str, np.ndarray]): Dictionary of parameters.
        """
        if "W" in params:
            self.W = params["W"]
        if "b" in params:
            self.b = params["b"]
