from typing import Any

import numpy as np

from src.layers.base import BaseLayer


class PositionalEncoding(BaseLayer):
    """
    Positional Encoding as described in the "Attention Is All You Need" paper
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        # Input validation
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError("max_len must be a positive integer")

        self.d_model = d_model
        self.max_len = max_len
        self.pe = self.build_pe(max_len, d_model)

    def build_pe(self, max_len: int, d_model: int):
        position = np.arange(max_len)[:, np.newaxis]
        # This is the term from 3.5: sin(pos/1000^{2i/d_model})
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def forward(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Add positional encoding to the input.

        Parameters:
            x (np.ndarray): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            np.ndarray: Tensor with positional encodings added.
        """
        # (batch_size, sequence_length, d_model)
        seq_len = x.shape[1]
        pe_slice = self.pe[:seq_len]
        # (1, max/seq_length, d_model) to fit above
        return x + pe_slice[np.newaxis, :, :]
