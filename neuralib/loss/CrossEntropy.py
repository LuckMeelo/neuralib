##
# EPITECH PROJECT, 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# CrossEntropy
##

from neuralib.constants import LossID
import numpy as np
from .ILoss import ALoss


class CategoricalCrossEntropy(ALoss):
    def __init__(self, eps=1e-12):
        super().__init__(LossID.CROSSENTROPY_CATEGORICAL)
        self.eps = eps  # Smoothing factor to avoid division by zero

    def forward(self, y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        # Calculate cross-entropy loss
        return -np.mean(y_true * np.log(y_pred))

    def backward(self, y_true, y_pred):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        # Gradient of cross-entropy loss
        return -(y_true / y_pred) / np.mean(y_true)


class BinaryCrossEntropy(ALoss):
    def __init__(self, eps=1e-12):
        super().__init__(LossID.CROSSENTROPY_BINARY)
        self.eps = eps

    def forward(self, y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        # Calculate binary cross-entropy loss
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        # Gradient of
