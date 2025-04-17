from typing import Any, Optional

from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.dropout import Dropout
from src.layers.normalization import LayerNorm


class ResidualConnection(BaseLayer):
    def __init__(
        self, normalized_shape: int, eps: Optional[float], dropout_rate: float = 0.0
    ):
        """
        Initializes ResidualConnection layer.

        Parameters:
            dropout_rate (float): Dropout rate for the layer.
        """
        super().__init__()
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNorm(normalized_shape=normalized_shape, eps=eps)

    def forward(self, x: ndarray, sublayer: Any):
        """
        Forward pass through the residual connection layer.

        Parameters:
            x (ndarray): Input data.
            sublayer (Any): The sublayer to apply the residual connection to.

        Returns:
            ndarray: Output data after applying the residual connection.
        """
        return x + self.dropout(sublayer(self.layer_norm(x)))
