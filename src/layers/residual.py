from typing import Optional

from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.dropout import Dropout
from src.layers.normalization import LayerNorm


class ResidualConnection(BaseLayer):
    def __init__(
        self, normalized_shape: int, eps: Optional[float], dropout_rate: float = 0.0
    ) -> None:
        """
        Initializes ResidualConnection layer.

        Parameters:
            dropout_rate (float): Dropout rate for the layer.
        """
        super().__init__()
        self.dropout = Dropout(dropout_rate)
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
