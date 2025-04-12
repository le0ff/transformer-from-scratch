from typing import Any, Dict, Optional

import numpy as np

from src.layers.base import BaseLayer


class Linear(BaseLayer):
    """
    Standard fully connected linear layer.
    """

    def __init__(
        self, input_dim: int, output_dim: int, seed: Optional[int] = None
    ) -> None:
        """
        Initializes Linear layer.

        Parameters:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            seed (Optional[int]): Seed for weight initialization.
        """
        super().__init__()

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Input and output dimensions must be positive integers.")

        if seed is None:
            rng = np.random.default_rng()
        else:
            if not isinstance(seed, int):
                raise ValueError("Seed must be an integer.")
            # Random number generator with a seed for reproducibility
            rng = np.random.default_rng(seed)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # He initialization for weights
        self.W = rng.standard_normal((input_dim, output_dim)) * np.sqrt(2.0 / input_dim)
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
            if params["W"].shape != (self.input_dim, self.output_dim):
                raise ValueError(
                    f"Expected W shape {(self.input_dim, self.output_dim)}, "
                    f"but got {params['W'].shape}"
                )
            self.W = params["W"]
        if "b" in params:
            if params["b"].shape != (self.output_dim,):
                raise ValueError(
                    f"Expected b shape {(self.output_dim,)}, "
                    f"but got {params['b'].shape}"
                )
            self.b = params["b"]

    # def backward(self, grad_output: np.ndarray) -> np.ndarray:
    #     """
    #     Computes gradients w.r.t. inputs and parameters.

    #     Parameters:
    #         grad_outputs (np.ndarray): Gradient of the loss w.r.t. the output of this layer.

    #     Returns:
    #         np.ndarray: Gradient of the loss w.r.t. the input of this layer.
    #     """
    #     if self._input_cache is None:
    #         raise ValueError("No input cache found. Forward pass must be called first.")

    #     # Compute gradients w.r.t. parameters (weights and bias)
    #     self.dW = self._input_cache.T @ grad_output
    #     self.db = np.sum(grad_output, axis=0)
    #     # Gradient w.r.t. input
    #     grad_inputs = grad_output @ self.W.T
    #     return grad_inputs
