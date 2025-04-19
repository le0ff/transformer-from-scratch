from typing import Dict

from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.normalization import LayerNorm


class Encoder(BaseLayer):
    """
    Encoder for the Transformer model.

    Consists of multiple encoder blocks, each containing a multi-head self-attention layer, feedforward block, residual connections, and dropout.
    """

    def __init__(self, layers: Dict[str, BaseLayer]) -> None:
        """
        Initializes the Encoder.

        Parameters:
            layers (Dict[str, BaseLayer]): Dictionary of layers in the encoder (in sequential order).
        """
        super().__init__()
        # dictionary of encoder blocks, e.g. {"encoder_block_1": EncoderBlock(...), ...}
        self.layers = layers
        # LayerNormalization, using the d_model of the first encoder block
        self.norm = LayerNorm(layers.values()[0].d_model)

    def forward(self, x: ndarray, mask: ndarray) -> ndarray:
        """
        Forward pass through the encoder.

        Parameters:
            x (ndarray): Input data.
            mask (ndarray): Mask for the input data.

        Returns:
            ndarray: Output data after passing through the encoder.
        """
        for layer in self.layers.values():
            x = layer(x, mask)
        return self.norm(x)

    def train(self) -> None:
        """Set the layer to training mode."""
        super().train()
        for layer in self.layers.values():
            layer.train()
        self.norm.train()

    def eval(self) -> None:
        """Set the layer to evaluation mode."""
        super().eval()
        for layer in self.layers.values():
            layer.eval()
        self.norm.eval()

    def get_parameters(self) -> Dict[str, ndarray]:
        """Get all parameters from the encoder."""
        # TODO: impmenet this method properly
        pass

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """Set parameters for the encoder."""
        # TODO: implement this method properly
        pass
