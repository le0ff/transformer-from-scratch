from typing import Dict, Optional

import numpy as np
from numpy import ndarray

from src.layers.activations.softmax import Softmax
from src.layers.base import BaseLayer
from src.layers.dropout import Dropout
from src.layers.linear import Linear


class MultiHeadAttentionBlock(BaseLayer):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_rate: float,
        seed: Optional[int] = None,
    ) -> None:
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
        if seed is not None and not isinstance(seed, int):
            raise ValueError("Seed must be an integer.")

        self.head_dim = d_model // n_heads

        if seed is not None:
            main_rng = np.random.default_rng(seed)
            max_seed_val = 2**31 - 1
            seeds = main_rng.integers(0, max_seed_val, size=5)
            seed_w_q = seeds[0]
            seed_w_k = seeds[1]
            seed_w_v = seeds[2]
            seed_w_o = seeds[3]
            seed_dropout = seeds[4]
        else:
            seed_w_q = None
            seed_w_k = None
            seed_w_v = None
            seed_w_o = None
            seed_dropout = None

        # Initialize weights for query, key, value, and output
        self.w_q = Linear(d_model, d_model, use_bias=False, seed=seed_w_q)
        self.w_k = Linear(d_model, d_model, use_bias=False, seed=seed_w_k)
        self.w_v = Linear(d_model, d_model, use_bias=False, seed=seed_w_v)
        self.w_o = Linear(d_model, d_model, use_bias=False, seed=seed_w_o)

        self.dropout = Dropout(dropout_rate, seed=seed_dropout)
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
            ((query @ key.transpose(0, 1, 3, 2)) / np.sqrt(d_k)), causal_mask=mask
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
            x.shape[0], x.shape[2], self.n_heads * self.head_dim
        )
        return self.w_o(x)

    def __call__(
        self, query: ndarray, key: ndarray, value: ndarray, mask: ndarray
    ) -> ndarray:
        return self.forward(query, key, value, mask)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get the parameters of the layer.

        Returns:
            Dict[str, np.ndarray]: Dictionary of parameters.
        """
        params = {
            "w_q": self.w_q.get_parameters()["W"],
            "w_k": self.w_k.get_parameters()["W"],
            "w_v": self.w_v.get_parameters()["W"],
            "w_o": self.w_o.get_parameters()["W"],
        }
        return params

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """
        Set the parameters of the layer.

        Parameters:
            params (Dict[str, np.ndarray]): Dictionary of parameters.
        """
        # Check for valid length of the input dictionary
        if len(params) != 4:
            raise ValueError(
                "Expected 4 parameters: 'w_q', 'w_k', 'w_v', and 'w_o'.\n "
                "Got: " + str(list(params.keys()))
            )

        # Check the shapes of the parameters
        for key in ["w_q", "w_k", "w_v", "w_o"]:
            # Validate shape
            if params[key].shape != (self.d_model, self.d_model):
                raise ValueError(
                    f"Expected {key} shape {(self.d_model, self.d_model)}, "
                    f"but got {params[key].shape}"
                )

        # Set parameters
        self.w_q.set_parameters({"W": params["w_q"]})
        self.w_k.set_parameters({"W": params["w_k"]})
        self.w_v.set_parameters({"W": params["w_v"]})
        self.w_o.set_parameters({"W": params["w_o"]})
