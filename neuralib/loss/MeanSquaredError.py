##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# MeanSquaredError
##

import numpy as np
from neuralib.constants import LossID
from .ILoss import ALoss


class MSE(ALoss):
    def __init__(self):
        super().__init__(LossID.MEAN_SQUARED_ERROR)

    def forward(self, y_true, y_pred):
        # Calculate mean squared error
        return np.mean(np.square(y_true - y_pred))

    def backward(self, y_true, y_pred):
        # Gradient of MSE (2 * (y_true - y_pred))
        return 2 * (y_true - y_pred) / np.mean(np.square(y_true - y_pred) + 1e-10)
