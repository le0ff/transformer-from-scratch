from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray

from src.layers.base import BaseLayer
from src.layers.linear import Linear


class ProjectionLayer(BaseLayer):
    """
    Projects output of transformer block into a probability distribution over the vocabulary.

    """

    def __init__(
        self, d_model: int, vocab_size: int, seed: Optional[int] = None
    ) -> None:
        """
        Initialize the projection layer.

        Parameters:
        d_model (int): Dimensionality of the model.
        vocab_size (int): Size of the vocabulary.
        seed (int, optional): Random seed for initialization.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = Linear(d_model, vocab_size, use_bias=True, seed=seed)

    def forward(self, x: ndarray, **kwargs: Any) -> ndarray:
        """
        Forward pass of the projection layer: Linear projection + log-softmax.

        Parameters:
        x (ndarray): Input data (output from transformer block) (..., d_model).

        Returns:
        ndarray: Log-probabilities (..., vocab_size).
        """
        logits = self.linear(x)

        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp
        return log_probs

    def train(self) -> None:
        """Set the layer to training mode."""
        super().train()
        self.linear.train()

    def eval(self) -> None:
        """Set the layer to evaluation mode."""
        super().eval()
        self.linear.eval()

    def get_parameters(self) -> Dict[str, ndarray]:
        """Get parameters of the projection layer."""
        return {"linear_" + k: v for k, v in self.linear.get_parameters().items()}

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """Set parameters of the projection layer."""
        if not params:
            raise ValueError("No parameters provided for projection layer.")

        # expects keys prefixed with 'linear_', e.g. 'linear_W', 'linear_b'
        linear_params = {}

        for key, value in params.items():
            if key.startswith("linear_"):
                param_name = key[len("linear_") :]
                linear_params[param_name] = value
            else:
                raise ValueError(f"Unexpected parameter key for ProjectionLayer: {key}")

        if linear_params:
            self.linear.set_parameters(linear_params)
