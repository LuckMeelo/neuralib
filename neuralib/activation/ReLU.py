##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# ReLU
##

import numpy as np
from neuralib.constants import ActivationID
from .IActivation import AActivation


class ReLU(AActivation):
    def __init__(self):
        super().__init__(ActivationID.RELU)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, 1, 0)


class LeakyReLU(AActivation):
    def __init__(self, alpha=0.01):  # Leaky ReLU parameter
        super().__init__(ActivationID.LEAKY_RELU)
        self.alpha = alpha

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(self.alpha * X, X)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, 1, self.alpha)


class ClippedReLU(AActivation):
    def __init__(self, clip_max=10):  # Maximum threshold
        super().__init__(ActivationID.CLIPPED_RELU)
        self.clip_max = clip_max

    def forward(self, X):
        return np.clip(X, 0, self.clip_max)  # Clip between 0 and clip_max

    def derivative(self, X):
        return np.where(X > 0, 1, 0)  # Same as ReLU for positive values
