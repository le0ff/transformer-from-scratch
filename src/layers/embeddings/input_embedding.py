from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray

from src.layers.base import BaseLayer


class InputEmbedding(BaseLayer):
    """
    Input Embedding Layer for Transformer.

    Maps token IDs to dense vectors and scales them by sqrt(d_model).
    Positional encoding is applied after this layer.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the embedding layer with scaled random weights.

        Parameters:
            d_model (int): The dimensionality of the embeddings.
            vocab_size (int): The number of tokens in the vocabulary.
            seed (Optional[int]): Optional random seed for reproducibility.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        scale_std = np.sqrt(1.0 / d_model)
        if seed is None:
            rng = np.random.default_rng()
        else:
            if not isinstance(seed, int):
                raise ValueError("Seed must be an integer.")
            # Random number generator with a seed for reproducibility
            rng = np.random.default_rng(seed)
        self.W_embed = (
            rng.standard_normal((vocab_size, d_model)).astype(np.float32) * scale_std
        )
        self.embedding_grad = np.zeros_like(self.W_embed)
        self.scale = np.sqrt(np.float32(d_model))
        self._input_cache = None

    def forward(self, x: ndarray, **kwargs: Any) -> ndarray:
        """
        Look up token embeddings and apply √d_model scaling.

        Parameters
        ----------
        x : ndarray
            Integer token-ID matrix of shape (batch_size, seq_len).

        Returns
        -------
        ndarray
            Embedded tokens of shape (batch_size, seq_len, d_model),
            multiplied by √d_model as described in Attention Is All You Need
            3.4 Embedding and Softmax
        """
        if x.ndim != 2:
            raise ValueError(
                f"Input x must have 2 dimensions (batch_size, sequence_length), but got {x.ndim}"
            )
        # Check Tokenizer Class
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("Input tensor must contain integer token indices.")
        if np.max(x) >= self.vocab_size or np.min(x) < 0:
            max_val, min_val = np.max(x), np.min(x)
            raise ValueError(
                f"Input token IDs out of bounds (0 to {self.vocab_size - 1}). "
                f"Found min={min_val}, max={max_val}. Check tokenizer/vocab size."
            )

        self._input_cache = x

        # Vectorised embedding lookup:
        # x has shape (batch_size, seq_len) with integer token IDs.
        # W_embed[x] returns the corresponding embedding vectors,
        # producing shape (batch_size, seq_len, d_model).

        token_embeds = self.W_embed[x]

        scaled_token_embeds = token_embeds * self.scale

        return scaled_token_embeds

    # def backward(self, d_out: ndarray):
    #     """
    #     Backpropagate gradients through the embedding layer.

    #     Args:
    #         d_out (ndarray): Gradient of loss w.r.t. output (batch, seq_len, d_model)
    #     """
    #     batch_size, seq_len = self._input_cache.shape
    #     for b in range(batch_size):
    #         for t in range(seq_len):
    #             token_id = self._input_cache[b, t]
    #             self.embedding_grad[token_id] += d_out[b, t]

    # def backward2(self, grad_output: ndarray) -> Dict[str, ndarray]:
    #     """
    #     Computes the gradient of the loss with respect to the embedding matrix.

    #     Args:
    #         grad_output (ndarray): Gradient of the loss with respect to the output
    #                                   of this layer. Shape: (batch_size, seq_len, d_model)

    #     Returns:
    #         Dict[str, ndarray]: Dictionary containing gradients for parameters.
    #                                {'W_embed': gradient_wrt_W_embed}
    #     """
    #     if not hasattr(self, "_input_cache"):
    #         raise RuntimeError("Need to call forward pass before backward.")

    #     batch_size, seq_len, _ = grad_output.shape
    #     dL_dW_embed = np.zeros_like(self.W_embed, dtype=np.float32)

    #     grad_output_scaled = grad_output * self.scale

    #     # Efficient scatter-add of gradients
    #     np.add.at(dL_dW_embed, self._input_cache, grad_output_scaled)

    #     return {"W_embed": dL_dW_embed}

    def get_parameters(self) -> Dict[str, ndarray]:
        """
        Returns the learnable parameters of the layer (the embedding matrix).
        """
        return {"W_embed": self.W_embed}

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """
        Set the learnable parameters of the layer (the embedding matrix).
        """
        if "W_embed" not in params:
            raise ValueError("Missing parameter: 'W_embed'")

        if params["W_embed"].shape != (self.vocab_size, self.d_model):
            raise ValueError(
                f"Shape mismatch for W_embed: expected {(self.vocab_size, self.d_model)}, "
                f"got {params['W_embed'].shape}"
            )

        self.W_embed = params["W_embed"]
