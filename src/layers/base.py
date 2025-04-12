from abc import ABC, abstractmethod
from typing import Any, Dict

from numpy import ndarray


class BaseLayer(ABC):
    """Abstract base class for all layers."""

    def __init__(self):
        self.training = True

    def train(self) -> None:
        """Set layer to training mode."""
        self.training = True

    def eval(self) -> None:
        """Set layer to evaluation mode."""
        self.training = False

    @abstractmethod
    def forward(self, x: ndarray, **kwargs: Any) -> ndarray:
        """
        Forward pass through the layer.

        Parameters:
            x (ndarray): Input data.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ndarray: Output data.
        """
        pass

    def get_parameters(self) -> Dict[str, ndarray]:
        """
        Get the parameters of the layer.
        Subclasses with parameters should override this method.

        Returns:
            Dict[str, ndarray]: Dictionary of parameters.
        """
        return {}

    def set_parameters(self, params: Dict[str, ndarray]) -> None:
        """
        Set the parameters of the layer.
        Subclasses with parameters should override this method.

        Parameters:
            params (Dict[str, ndarray]): Dictionary of parameters {name: value}.
        """
        pass

    def __call__(self, x: ndarray, **kwargs: Any) -> ndarray:
        """
        Call the forward method of the layer.

        Parameters:
            x (ndarray): Input data.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ndarray: Output data.
        """
        return self.forward(x, **kwargs)

    # @abstractmethod
    # def backward(self, output_grad: ndarray) -> ndarray:
    #     """
    #     Backward pass through the layer.

    #     Parameters:
    #         output_grad (ndarray): Gradient of the loss with respect to the output.

    #     Returns:
    #         ndarray: Gradient of the loss with respect to the input.
    #     """
    #     pass
