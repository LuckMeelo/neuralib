##
# EPITECH PROJECT, 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# SoftMax
##

from neuralib.constants import ActivationID
from .IActivation import AActivation
import numpy as np


class Softmax(AActivation):
    def __init__(self):
        super().__init__(ActivationID.SOFTMAX)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # For numerical stability
        exp_X = np.exp(X - X.max(axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        # Softmax derivative is a bit more complex, involving the output itself
        probs = self.forward(X)
        return probs[:, np.newaxis] * (np.identity(probs.shape[1]) - probs)
