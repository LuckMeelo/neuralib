##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# CrossEntropy
##

import numpy as np
from neuralib.constants import LossID
from .ILoss import ALoss


class CategoricalCrossEntropy(ALoss):
    # TODO review formulas
    def __init__(self, eps=1e-12):
        super().__init__(LossID.CROSSENTROPY_CATEGORICAL)
        self.eps = eps  # Smoothing factor to avoid division by zero

    # nnfs version
    # def forward(self, y_pred, y_true): 
 
    #     # Number of samples in a batch 
    #     samples = len(y_pred) 
 
    #     # Clip data to prevent division by 0 
    #     # Clip both sides to not drag mean towards any value 
    #     y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) 
 
    #     # Probabilities for target values - 
    #     # only if categorical labels 
    #     if len(y_true.shape) == 1: 
    #         correct_confidences = y_pred_clipped[ 
    #             range(samples), 
    #             y_true 
    #         ] 
 
    #     # Mask values - only for one-hot encoded labels 
    #     elif len(y_true.shape) == 2: 
    #         correct_confidences = np.sum( 
    #             y_pred_clipped * y_true, 
    #             axis=1 
    #         ) 
 
    #     # Losses 
    #     negative_log_likelihoods = -np.log(correct_confidences) 
    #     return negative_log_likelihoods 
    
    def forward(self, y_e, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        # Calculate cross-entropy loss
        return - np.sum(y_e * np.log(y_pred + self.eps), axis=1)

    def backward(self, y_e, y_pred):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, self.eps, 1. - self.eps)
        # Gradient of cross-entropy loss
        return -y_e / (y_pred + self.eps)


class BinaryCrossEntropy(ALoss):
    # TODO review formulas
    def __init__(self, eps=1e-12):
        super().__init__(LossID.CROSSENTROPY_BINARY)
        self.eps = eps

    def forward(self, y_e, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        # Calculate binary cross-entropy loss
        return -np.mean(y_e * np.log(y_pred) + (1.0 - y_e) * np.log(1.0 - y_pred))

    def backward(self, y_e, y_pred):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        # Gradient of binary cross-entropy loss
        return -(y_e / y_pred) - ((1.0 - y_e) / (1.0 - y_pred))
