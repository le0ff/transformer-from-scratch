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
        self._mask_cache = None

    def forward(self, x: np.ndarray, causal_mask: np.ndarray = None) -> np.ndarray:
        """
        Apply the softmax function to the input array.

        Parameters:
        x (np.ndarray): Input data (logits).
        causal_mask (np.ndarray): Mask to apply to the input data.

        Returns:
        np.ndarray: Softmax probabilities.
        """
        self._input_cache = x

        if causal_mask is not None:
            if not x.shape == causal_mask.shape:
                raise ValueError("Input and mask must have the same shape.")
            self._mask_cache = causal_mask
            # Apply mask
            x = np.where(causal_mask, x, -1e9)

        # Subtract max for numerical stability
        shift_x = x - np.max(x, axis=self.axis, keepdims=True)
        e_x = np.exp(shift_x)
        sum_e_x = np.sum(e_x, axis=self.axis, keepdims=True)
        return e_x / sum_e_x

    # def backward(self, grad_output: np.ndarray) -> np.ndarray:
    # """
    # Backward pass of softmax.

    # Args:
    #     grad_output (np.ndarray): Gradient of the loss with respect to the output.

    # Returns:
    #     np.ndarray: Gradient w.r.t input logits.
    # """
    # x = self._input_cache
    # mask = self._mask_cache

    # if mask is not None:
    #     x = np.where(mask, x, -1e9)

    # # Recompute softmax
    # x_max = np.max(x, axis=self.axis, keepdims=True)
    # exps = np.exp(x - x_max)
    # sums = np.sum(exps, axis=self.axis, keepdims=True)
    # softmax_out = exps / sums

    # dx = np.zeros_like(grad_output)

    # # Iterate over batch dimensions to apply Jacobian-vector product
    # it = np.nditer(softmax_out[..., 0], flags=["multi_index"])
    # while not it.finished:
    #     idx = it.multi_index
    #     s = softmax_out[idx]  # softmax vector
    #     d = grad_output[idx]  # upstream gradient

    #     if s.ndim == 0:
    #         dx[idx] = s * (1 - s) * d
    #     else:
    #         s = s.reshape(-1, 1)
    #         jacobian = np.diagflat(s) - s @ s.T
    #         dx[idx] = jacobian @ d

    #     it.iternext()

    # return dx
