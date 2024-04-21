##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# Sigmoid
##

import numpy as np
from neuralib.constants import ActivationID
from .IActivation import AActivation


class Sigmoid(AActivation):
    def __init__(self):
        super().__init__(ActivationID.SIGMOID)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X) * (1 - self.forward(X))
