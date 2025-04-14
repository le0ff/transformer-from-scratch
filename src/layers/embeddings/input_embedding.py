from typing import Any, Dict, Optional

import numpy as np

from src.layers.base import BaseLayer


class InputEmbedding(BaseLayer):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        scale = np.sqrt(1.0 / d_model)
        if seed is None:
            rng = np.random.default_rng()
        else:
            if not isinstance(seed, int):
                raise ValueError("Seed must be an integer.")
            # Random number generator with a seed for reproducibility
            rng = np.random.default_rng(seed)
        self.W_embed = (
            rng.random.standard_normal((vocab_size, d_model)).astype(np.float32) * scale
        )
        self.embedding_grad = np.zeros_like(self.embedding_matrix)
        self.scale = np.sqrt(np.float32(d_model))

    def forward(self, x: np.ndarry, **kwargs: Any) -> np.ndarray:
        """
        Tokenize the input text, apply embedding and positional encoding.

        Returns:
            np.ndarray: Normalize embeddings
        """
        if x.ndim != 2:
            raise ValueError(
                f"Input x must have 2 dimensions (batch_size, sequence_length), but got {x.ndim}"
            )
        if np.max(x) >= self.vocab_size or np.min(x) < 0:
            max_val, min_val = np.max(x), np.min(x)
            raise ValueError(
                f"Input token IDs out of bounds (0 to {self.vocab_size - 1}). "
                f"Found min={min_val}, max={max_val}. Check tokenizer/vocab size."
            )

        # Token Embedding Lookup
        # Select rows from W_embed based on integer IDs in x
        token_embeds = self.W_embed[x]  # Shape: (batch_size, seq_len, d_model)

        # Scale token embeddings
        scaled_token_embeds = token_embeds * self.scale_factor

        return scaled_token_embeds

        # return self.embedding(x) * np.sqrt(self.d_model)

    # def backward(self, d_out: np.ndarray):
    #     """
    #     Backpropagate gradients through the embedding layer.

    #     Args:
    #     d_out (np.ndarray): Gradient of loss w.r.t. output (batch, seq_len, d_model)
    #     """
    #     batch_size, seq_len = self.token_ids.shape
    #     grad = d_out

    #     for b in range(batch_size):
    #         for t in range(seq_len):
    #             token_id = self.token_ids[b, t]
    #             self.embedding_grad[token_id] += grad[b, t]

    # def backward2(self, grad_output: np.ndarray) -> Dict[str, np.ndarray]:
    #     """
    #     Computes the gradient of the loss with respect to the embedding matrix.

    #     Args:
    #         grad_output (np.ndarray): Gradient of the loss with respect to the output
    #                         of this layer. Shape: (batch_size, seq_len, d_model).

    #     Returns:
    #         Dict[str, np.ndarray]: Dictionary containing gradients for parameters.
    #                             {'W_embed': gradient_wrt_W_embed}
    #     """
    #     # We need the input `x` that generated the output `y`
    #     # Store `x` during the forward pass: self.x = x
    #     if not hasattr(self, "x"):
    #         raise RuntimeError("Need to call forward pass first to store input x.")

    #     batch_size, seq_len, _ = grad_output.shape
    #     # Initialize gradient matrix for W_embed with zeros
    #     dL_dW_embed = np.zeros_like(self.W_embed, dtype=np.float32)

    #     # Apply the scaling factor from the incoming gradient
    #     grad_output_scaled_contribution = grad_output * self.scale_factor

    # Efficiently add gradients:
    # For each token_id present in self.x, sum the corresponding
    # incoming gradients (grad_output_scaled_contribution) at the positions where it occurred.
    # np.add.at performs this scatter-add operation efficiently.
    # It iterates through self.x, and for each element x[b, s] (which is a token_id),
    # it adds the corresponding vector grad_output_scaled_contribution[b, s, :]
    # to the row dL_dW_embed[token_id, :].
    # np.add.at(dL_dW_embed, self.x, grad_output_scaled_contribution)

    # # Store the input x in the forward pass
    # # del self.x # Clean up stored input

    # return {"W_embed": dL_dW_embed}

    # def step(self, lr=1e-3):
    # self.embedding_matrix -= lr * self.embedding_grad

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Returns the learnable parameters of the layer (the embedding matrix).
        """
        return {"W_embed": self.W_embed}

    def set_parameters(self, params: Dict[str, np.ndarray]):
        """
        Set the learnable parameters of the layer (the embedding matrix).
        """
        if "W_embed" in params:
            # Validate shape of W_embed
            if params["W_embed"].shape != (self.vocab_size, self.d_model):
                raise ValueError(
                    f"Shape mismatch for W_embed: expected {self.vocab_size, self.d_model}, "
                    f"but got {params['W_embed'].shape}"
                )
        self.W_embed = params["W_embed"]
