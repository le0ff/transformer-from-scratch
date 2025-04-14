from typing import Any, Dict, Optional

import numpy as np

from src.layers.base import BaseLayer


class Linear(BaseLayer):
    """
    Standard fully connected linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True,
        seed: Optional[int] = None,
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
        self.use_bias = use_bias

        # He initialization for weights
        self.W = rng.standard_normal((input_dim, output_dim)) * np.sqrt(2.0 / input_dim)

        # initialize bias to zero if use_bias set to true
        if use_bias:
            self.b = np.zeros(output_dim)
        else:
            self.b = None

        # Cache for backward pass
        self._input_cache = None
        # self.dW = None
        # self.db = None

    def forward(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Forward pass through the layer.

        Parameters:
            x (np.ndarray): Input data.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            np.ndarray: Output data.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input shape with last dimension {self.input_dim}, "
                f"but got {x.shape[-1]}"
            )
        # store input for backward pass
        self._input_cache = x

        output = x @ self.W
        # if use_bias is True, add bias
        if self.use_bias:
            assert self.b is not None, "Bias should be initialized to zero"
            output = output + self.b
        return output

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get the parameters of the layer.

        Returns:
            Dict[str, np.ndarray]: Dictionary of parameters.
        """
        params = {"W": self.W}
        if self.use_bias:
            assert self.b is not None, "Bias is enabled but not initialized"
            params["b"] = self.b
        return params

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """
        Set the parameters of the layer.

        Parameters:
            params (Dict[str, np.ndarray]): Dictionary of parameters.
        """
        # Set weights
        if "W" in params:
            # Validate shape of W
            if params["W"].shape != (self.input_dim, self.output_dim):
                raise ValueError(
                    f"Expected W shape {(self.input_dim, self.output_dim)}, "
                    f"but got {params['W'].shape}"
                )
            self.W = params["W"]
        else:
            raise ValueError("Weight parameter 'W' missing in params dictionary.")

        # Set bias (if applicable)
        if self.use_bias:
            if "b" in params:
                # Validate shape of b
                if params["b"].shape != (self.output_dim,):
                    raise ValueError(
                        f"Expected b shape {(self.output_dim,)}, "
                        f"but got {params['b'].shape}"
                    )
                self.b = params["b"]
            else:
                raise ValueError(
                    "Bias parameter 'b' missing in params dictionary, but use_bias is True."
                )
        else:
            # If bias is not used, ensure 'b' is not provided or set self.b to None
            if "b" in params:
                print(
                    "Warning: Provided bias 'b' in params when use_bias is False. Ignoring."
                )
            self.b = None

    # def backward(self, grad_output: np.ndarray) -> np.ndarray:
    #     """
    #     Computes gradients w.r.t. inputs and parameters.

    #     Parameters:
    #         grad_output (np.ndarray): Gradient of the loss w.r.t. the output of this layer.

    #     Returns:
    #         np.ndarray: Gradient of the loss w.r.t. the input of this layer.
    #     """
    #     if self._input_cache is None:
    #         raise ValueError("No input cache found. Forward pass must be called first.")

    #     # Shape: (N, ..., input_dim)
    #     x = self._input_cache

    #     # Validate shape of grad_output
    #     if grad_output.shape[-1] != self.output_dim:
    #         raise ValueError(
    #             f"Expected grad_output shape with last dimension {self.output_dim}, "
    #             f"but got {grad_output.shape[-1]}"
    #         )
    #     # Validate leading dimensions
    #     if x.shape[:-1] != grad_output.shape[:-1]:
    #         raise ValueError(
    #             f"Leading dimensions of grad_output {grad_output.shape[:-1]} "
    #             f"do not match cached input's leading dimensions {x.shape[:-1]}."
    #         )

    #     # Gradient w.r.t. input
    #     grad_input = grad_output @ self.W.T

    #     # Compute gradients w.r.t. weights
    #     # calculate total number of samples (e.g., (batch_size, input_dim) -> batch_size, (batch_size, seq_len, input_dim) -> batch_size * seq_len)
    #     num_samples = np.prod(x.shape[:-1])
    #     x_reshaped = x.reshape(num_samples, self.input_dim)
    #     grad_output_reshaped = grad_output.reshape(num_samples, self.output_dim)
    #     self.dW = x_reshaped.T @ grad_output_reshaped

    #     if self.use_bias:
    #         assert self.b is not None, "Bias should be initialized to zero"
    #         # Gradient w.r.t. bias
    #         # sum grad output along all axes except last one (e.g., (0,) for 2D, (0, 1) for 3D)
    #         sum_axes = tuple(range(grad_output.ndim - 1))
    #         self.db = np.sum(grad_output, axis=sum_axes)
    #         # Ensure calculated db has the correct shape (D_out,)
    #         assert self.db.shape == (self.output_dim,), (
    #             f"Calculated db shape {self.db.shape} doesn't match expected {(self.output_dim,)}"
    #         )
    #     else:
    #         # If no bias, set db to None
    #         self.db = None

    #     return grad_input
