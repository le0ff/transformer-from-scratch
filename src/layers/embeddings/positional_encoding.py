from typing import Any, Optional

import numpy as np
from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.dropout import Dropout


class PositionalEncoding(BaseLayer):
    """
    Positional Encoding as described in the "Attention Is All You Need" paper
    """

    def __init__(
        self, d_model: int, max_len: int, dropout_rate, seed: Optional[int] = None
    ) -> None:
        """
        Initialize the PositionalEncoding layer.

        Parameters:
            d_model (int): The dimension of the model (embedding size).
            max_len (int): The maximum length of the sequence for which to create encodings.
            dropout_rate (float): Dropout rate to apply to the positional encodings.
            seed (Optional[int]): Optional random seed for dropout reproducibility.
        """
        super().__init__()
        # Input validation
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError("max_len must be a positive integer")
        # Validate dropout rate
        if not isinstance(dropout_rate, (int, float)) or not (
            0.0 <= dropout_rate < 1.0
        ):
            raise ValueError("dropout_rate must be a float in [0.0, 1.0).")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("Seed must be an integer or None.")

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = Dropout(dropout_rate, seed)
        self.pe = self.build_pe(max_len, d_model)

    def build_pe(self, max_len: int, d_model: int) -> ndarray:
        """
        Build the sinusoidal positional encoding matrix.

        Parameters:
            max_len (int): The maximum length of the sequence.
            d_model (int): The dimension of the model (embedding size).

        Returns:
            ndarray: A matrix of shape (max_len, d_model) containing the positional encodings.
        """
        position = np.arange(max_len)[:, np.newaxis]
        # This is the term from 3.5: sin(pos/1000^{2i/d_model})
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        # Trims in case of odd values for d_model
        # pe[:, 1::2] = np.cos(position * div_term[: (d_model // 2)])
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def forward(self, x: ndarray, **kwargs: Any) -> ndarray:
        """
        Add positional encoding to the input.

        Parameters:
            x (ndarray): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            ndarray: Tensor with positional encodings added.
        """
        batch_size, seq_len, input_d_model = x.shape

        if input_d_model != self.d_model:
            raise ValueError(
                f"Input feature dimension {input_d_model} does not match layer's d_model {self.d_model}"
            )
        if seq_len > self.max_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds layer's pre-computed max_len {self.max_len}"
            )

        pe_slice = self.pe[:seq_len]
        # (1, seq_length, d_model) to fit above
        out = x + pe_slice[np.newaxis, :, :]
        # Apply dropout
        return self.dropout(out, **kwargs)

    def train(self) -> None:
        """Set the layer to training mode."""
        super().train()
        self.dropout.train()

    def eval(self) -> None:
        """Set the layer to evaluation mode."""
        super().eval()
        self.dropout.eval()
