##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# __init__
##


from neuralib.constants import LossID
from abc import ABC, abstractmethod
import numpy as np


class ILoss(ABC):

    @abstractmethod
    def get_id(self) -> LossID:
        pass

    @abstractmethod
    def forward(self, y_e: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the loss between the true labels and predictions.

        Args:
            y_e: The ground truth labels, typically a NumPy array.
            y_pred: The model's predictions, typically a NumPy array.

        Returns:
            A float representing the calculated loss value.
        """
        pass

    @abstractmethod
    def backward(self, y_: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the loss with respect to the predictions.

        This is used for backpropagation during training.

        Args:
            y_e: The ground truth labels, typically a NumPy array.
            y_pred: The model's predictions, typically a NumPy array.

        Returns:
            A NumPy array representing the gradient of the loss w.r.t. y_pred.
        """
        pass


class ALoss(ILoss):
    def __init__(self, id: LossID):
        super().__init__()  # Call parent class constructor
        self.id = id

    def get_id(self) -> LossID:
        return (self.id)

    def __call__(self, y_e: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the loss between the true labels and predictions.

        Args:
            y_e: The ground truth labels, typically a NumPy array.
            y_pred: The model's predictions, typically a NumPy array.

        Returns:
            A float representing the calculated loss value.
        """
        return self.forward(y_e, y_pred)  # By default, call forward
