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

    def forward(self, y_e, y_pred):
        axis = 1
        if (y_pred.ndim == 1):
            axis = None
        # Calculate mean squared error
        return np.mean(np.square(y_e - y_pred), axis=axis)

    def backward(self, y_e, y_pred):
        # Gradient of MSE (2 * (y_e - y_pred))
        return 2 * (y_e - y_pred) / len(y_pred)
