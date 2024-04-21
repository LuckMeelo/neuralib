##
# EPITECH PROJECT, 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# __init__
##

from neuralib.constants import ActivationID
from abc import ABC, abstractmethod
import numpy as np


class IActivation(ABC):

    @abstractmethod
    def get_id(self) -> ActivationID:
        pass

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Applies the activation function to the input data.

        Args:
            X: The input data, typically a NumPy array.

        Returns:
            The output of the activation function applied to X.
        """
        pass

    @abstractmethod
    def derivative(self, X: np.ndarray) -> np.ndarray:
        """Calculates the derivative of the activation function at the input.

        Args:
            X: The input data, typically a NumPy array.

        Returns:
            The derivative of the activation function evaluated at X.
        """
        pass


class AActivation(IActivation):
    def __init__(self, id: ActivationID):
        super().__init__()  # Call parent class constructor
        self.id = id

    def get_id(self) -> ActivationID:
        return (self.id)

    def __call__(self, X: np.ndarray):
        """Applies the activation function to the input data.

        This method is implemented here to make instances of AActivation callable.

        Args:
            X: The input data, typically a NumPy array.

        Returns:
            The output of the activation function applied to X.
        """
        return self.forward(X)  # By default, call forward
