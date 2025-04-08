from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseLayer(ABC):
    """
    Abstract base class for all layers.
    """

    def __init__(self):
        self.training = True

    def train(self) -> None:
        """
        Set layer to training mode.
        """
        self.training = True

    def eval(self) -> None:
        """
        Set layer to evaluation mode.
        """
        self.training = False

    @abstractmethod
    def forward(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Forward pass through the layer.

        Parameters:
            x (np.ndarray): Input data.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            np.ndarray: Output data.
        """
        pass

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get the parameters of the layer.

        Returns:
            Dict[str, np.ndarray]: Dictionary of parameters.
        """
        return {}

    def __call__(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Call the forward method of the layer.

        Parameters:
            x (np.ndarray): Input data.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            np.ndarray: Output data.
        """
        return self.forward(x, **kwargs)
