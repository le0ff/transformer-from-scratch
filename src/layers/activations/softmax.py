import numpy as np

from src.layers.base import BaseLayer


class Softmax(BaseLayer):
    """
    Numerically stable Softmax activation.
    """

    def __init__(self, axis: int = -1) -> None:
        """
        Initialize the Softmax activation function.

        Parameters:
        axis (int): The axis along which softmax will be computed.
        """
        super().__init__()
        self.axis = axis
        # Cache for backward pass
        self._input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the softmax function to the input array.

        Parameters:
        x (np.ndarray): Input data (logits).

        Returns:
        np.ndarray: Softmax probabilities.
        """
        # store input for backward pass
        self._input_cache = x
        # Subtract max for numerical stability
        shift_x = x - np.max(x, axis=self.axis, keepdims=True)
        e_x = np.exp(shift_x)
        sum_e_x = np.sum(e_x, axis=self.axis, keepdims=True)
        return e_x / sum_e_x
