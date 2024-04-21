##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# BinaryStep
##

import numpy as np
from neuralib.constants import ActivationID
from .IActivation import AActivation


class BinaryStep(AActivation):
    def __init__(self):
        super().__init__(ActivationID.BINARY_STEP)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X >= 0, 1, 0)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return np.zeros_like(X)  # Binary step has zero derivative
