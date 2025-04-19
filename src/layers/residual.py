from typing import Dict, Optional

from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.dropout import Dropout
from src.layers.normalization import LayerNorm


class ResidualConnection(BaseLayer):
    def __init__(
        self,
        normalized_shape: int,
        eps: Optional[float],
        dropout_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initializes ResidualConnection layer.

        Parameters:
            dropout_rate (float): Dropout rate for the layer.
        """
        super().__init__()

        if dropout_rate < 0.0 or dropout_rate > 1.0:
            raise ValueError("Dropout rate must be between 0 and 1.")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("Seed must be an integer.")

        self.dropout = Dropout(dropout_rate, seed=seed)
        self.layer_norm = LayerNorm(normalized_shape=normalized_shape, eps=eps)

    def forward(self, x: ndarray, sublayer: BaseLayer) -> ndarray:
        """
        Forward pass through the residual connection layer.

        Parameters:
            x (ndarray): Input data.
            sublayer (BaseLayer): The sublayer to apply the residual connection to.

        Returns:
            ndarray: Output data after applying the residual connection.
        """
        return x + self.dropout(sublayer(self.layer_norm(x)))

    def train(self) -> None:
        """Set layer to training mode."""
        super().train()
        self.dropout.train()
        self.layer_norm.train()

    def eval(self) -> None:
        """Set layer to evaluation mode."""
        super().eval()
        self.dropout.eval()
        self.layer_norm.eval()

    def get_parameters(self) -> Dict[str, ndarray]:
        """
        Get parameters of the ResidualConnection layer, namespaced by sublayer.
        Returns:
            dict: Dictionary with keys like 'layernorm_gamma', 'layernorm_beta'.
        """
        params = {}
        layer_norm_params = self.layer_norm.get_parameters()
        for name, param in layer_norm_params.items():
            params[f"layernorm_{name}"] = param

        return params

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """
        Set parameters for the ResidualConnection layer, expecting namespaced keys.
        Parameters:
            params (dict): Dictionary with keys like 'layernorm_gamma', 'layernorm_beta'.
        """
        layernorm_params = {}
        unexpected_keys = []
        for key, value in params.items():
            if key.startswith("layernorm_"):
                param_name = key[len("layernorm_") :]
                layernorm_params[param_name] = value
            else:
                unexpected_keys.append(key)
        if unexpected_keys:
            raise ValueError(
                f"Unexpected parameter keys for ResidualConnection: {unexpected_keys}"
            )
        if not layernorm_params:
            raise ValueError("No parameters found for LayerNorm in ResidualConnection.")
        # additonal check if layernorm_params has all required keys
        required = {"gamma", "beta"}
        missing = required - set(layernorm_params.keys())
        if missing:
            raise ValueError(f"Missing LayerNorm parameters: {missing}")
        self.layer_norm.set_parameters(layernorm_params)
