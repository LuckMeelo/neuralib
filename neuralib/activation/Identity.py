##
# PROJECT, 2024
# neuralib [WSL: Ubuntu-22.04]
# File description:
# Identity
##


import numpy as np
from neuralib.constants import ActivationID
from .IActivation import AActivation


class Identity(AActivation):
    def __init__(self):
        super().__init__(ActivationID.IDENTITY)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return np.ones_like(X)
