##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# SoftMax
##

import numpy as np
from neuralib.constants import ActivationID
from .IActivation import AActivation


class Softmax(AActivation):
    def __init__(self):
        super().__init__(ActivationID.SOFTMAX)

    def forward(self, X: np.ndarray) -> np.ndarray:
        axis = 1
        if (X.ndim == 1):
            axis = None
        exp_X = np.exp(X - np.max(X, axis=axis, keepdims=True))
        return exp_X / np.sum(exp_X, axis=axis, keepdims=True)

    def derivative(self, X: np.ndarray) -> np.ndarray:
        # Softmax derivative is a bit more complex, involving the output itself
        probs = self.forward(X)
        return probs[:, np.newaxis] * (np.identity(probs.shape[1]) - probs)
        # TODO review derivative
        # Create uninitialized array
        # self.dinputs = np.empty_like(X)

        # # Enumerate outputs and gradients
        # for index, (single_output, single_dvalues) in \
        #         enumerate(zip(self.output, X)):
        #     # Flatten output array
        #     single_output = single_output.reshape(-1, 1)
        #     # Calculate Jacobian matrix of the output and
        #     jacobian_matrix = np.diagflat(single_output) - \
        #         np.dot(single_output, single_output.T)
        #     # Calculate sample-wise gradient
        #     # and add it to the array of sample gradients
        #     self.dinputs[index] = np.dot(jacobian_matrix,
        #                                  single_dvalues)
