##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# Tanh
##

import numpy as np
from neuralib.constants import ActivationID
from .IActivation import AActivation


class Tanh(AActivation):
    def __init__(self):
        super().__init__(ActivationID.TANH)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(X) * np.tanh(X)
