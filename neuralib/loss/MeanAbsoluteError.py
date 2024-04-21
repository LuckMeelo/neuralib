##
# EPITECH PROJECT, 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# MeanAbsoluteError
##


from neuralib.constants import LossID
from .ILoss import ALoss
import numpy as np


class MAE(ALoss):
    def __init__(self):
        super().__init__(LossID.MEAN_ABOSOLUTE_ERROR)

    def forward(self, y_true, y_pred):
        # Calculate mean absolute error
        return np.mean(np.abs(y_true - y_pred))

    def backward(self, y_true, y_pred):
        # Gradient of MAE (sign function)
        # Avoid division by zero
        return np.sign(y_true - y_pred) / np.mean(np.abs(y_true - y_pred) + 1e-10)
