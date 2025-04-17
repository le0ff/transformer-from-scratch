from typing import Any, Dict, Optional

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
        for layer in self._layers.values():
            layer.train()

    def eval(self) -> None:
        """Set the layer to evaluation mode."""
        super().eval()
        for layer in self._layers.values():
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

    def get_parameters(self) -> Dict[str, ndarray]:
        """Gets parameters, prefixing keys with sub-layer names."""
        all_params = {}
        for layer_name, layer in self._layers.items():
            layer_params = layer.get_parameters()
            for param_name, param_value in layer_params.items():
                all_params[f"{layer_name}_{param_name}"] = param_value
        return all_params

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """Sets parameters, expecting prefixed keys."""
        params_by_layer = {
            name: {} for name, layer in self._layers.items() if layer.get_parameters()
        }
        processed_keys = set()

        for prefixed_key, param_value in params.items():
            for layer_name in params_by_layer.keys():
                prefix = f"{layer_name}_"
                if prefixed_key.startswith(prefix):
                    original_param_name = prefixed_key[len(prefix) :]
                    params_by_layer[layer_name][original_param_name] = param_value
                    processed_keys.add(prefixed_key)
                    break

        if len(processed_keys) != len(params):
            missing_keys = set(params.keys()) - processed_keys
            raise ValueError(f"Missing parameters for layers: {missing_keys}")

        for layer_name, layer_params_dict in params_by_layer.items():
            if layer_params_dict:
                try:
                    self._layers[layer_name].set_parameters(layer_params_dict)
                except ValueError as e:
                    raise ValueError(
                        f"Error setting parameters for sub-layer '{layer_name}': {e}"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"Unexpected error setting parameters for sub-layer '{layer_name}': {e}"
                    ) from e

    # def backward(self, grad_output: ndarray) -> ndarray:
    #     """
    #     Performs backward pass through the feedforward block.

    #     Computes the gradient of the loss with respect to the input of this block,
    #     by backpropagating the gradient through the layers in reverse order of
    #     their definition in self._sub_layers.

    #     Parameters:
    #         grad_output (ndarray): Gradient of the loss with respect to the output of this block.

    #     Returns:
    #         ndarray: Gradient of the loss with respect to the input of this block.
    #     """
    #     for layer in reversed(self._layers.values()):
    #         grad_output = layer.backward(grad_output)
    #     return grad_output
