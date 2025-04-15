import numpy as np
from numpy import ndarray

from src.layers.activations.softmax import Softmax
from src.layers.base import BaseLayer
from src.layers.dropout import Dropout
from src.layers.linear import Linear


class MultiHeadAttentionBlock(BaseLayer):
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float) -> None:
        """Initializes MultiHeadAttentionBlock layer.
        Parameters:
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        # Check for valid input parameters
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive integers.")
        if dropout_rate < 0.0 or dropout_rate > 1.0:
            raise ValueError("Dropout rate must be between 0 and 1.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.head_dim = d_model // n_heads

        # Initialize weights for query, key, value, and output
        self.w_q = Linear(d_model, d_model, use_bias=False)
        self.w_k = Linear(d_model, d_model, use_bias=False)
        self.w_v = Linear(d_model, d_model, use_bias=False)
        self.w_o = Linear(d_model, d_model, use_bias=False)

        self.dropout = Dropout(dropout_rate)
        self.softmax = Softmax()

    @staticmethod
    def attention(
        query: ndarray,
        key: ndarray,
        value: ndarray,
        mask: ndarray,
        dropout: Dropout,
        softmax: Softmax,
    ) -> ndarray:
        """Compute attention scores according to the Attention is all you need.
        Parameters:
            query (ndarray): Query matrix.
            key (ndarray): Key matrix.
            value (ndarray): Value matrix.
            mask (ndarray): Mask matrix.
            dropout (Dropout): Dropout layer.
            softmax (Softmax): Softmax layer.
        Returns:
            ndarray: Output matrix after applying attention.
        """
        # Check for valid input shapes
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or mask.ndim != 4:
            raise ValueError(
                "Input arrays must be 4-dimensional. Current shapes: "
                f"query: {query.shape}, key: {key.shape}, value: {value.shape}, mask: {mask.shape}"
            )
        d_k = query.shape[-1]

        # Compute attention scores
        attention_scores = softmax(
            (query @ key.transpose(0, 2, 1)) / np.sqrt(d_k), mask
        )

        # Apply dropout to attention scores
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value

    def forward(
        self, query: ndarray, key: ndarray, value: ndarray, mask: ndarray
    ) -> ndarray:
        """Forward pass for MultiHeadAttentionBlock.
        Even though in self attention the query, key and value inputs are the same, we keep them separate
        to allow for cross attention.
        This allows for more flexibility in the model architecture.
        Parameters:
            query (ndarray): Query input.
            key (ndarray): Key input.
            value (ndarray): Value input.
            mask (ndarray): Mask matrix.
        Returns:
            ndarray: Output of the attention block.
        """
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)
        # Reshape query, key, value for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, head_dim)
        query = query.reshape(
            query.shape[0], query.shape[1], self.n_heads, self.head_dim
        )
        key = key.reshape(key.shape[0], key.shape[1], self.n_heads, self.head_dim)
        value = value.reshape(
            value.shape[0], value.shape[1], self.n_heads, self.head_dim
        )
        # Transpose to get the correct shape for multi-head attention
        # (batch_size, n_heads, seq_len, head_dim)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        x = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout, self.softmax
        )

        # Reshape back to (batch_size, seq_len, d_model)
        x = x.transpose(0, 2, 1, 3).reshape(
            query.shape[0], query.shape[1], self.n_heads * self.head_dim
        )
        return self.w_o(x)
