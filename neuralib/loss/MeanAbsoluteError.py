##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# MeanAbsoluteError
##


import numpy as np
from neuralib.constants import LossID
from .ILoss import ALoss


class MAE(ALoss):
    def __init__(self):
        super().__init__(LossID.MEAN_ABOSOLUTE_ERROR)

    def forward(self, y_e, y_pred):
        axis = 1
        if (y_pred.ndim == 1):
            axis = None
        # Calculate mean absolute error
        return np.mean(np.abs(y_e - y_pred), axis=axis)

    def backward(self, y_e, y_pred):
        # Gradient of MAE (sign function)
        return np.sign(y_pred - y_e) / len(y_pred)
