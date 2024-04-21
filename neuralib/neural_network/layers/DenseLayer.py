##
# Project 2024
# neural_test [WSL : Ubuntu-22.04]
# File description:
# Denselayer
##

import numpy as np
from neuralib.activation import IActivation
from .ILayer import ALayer


class DenseLayer(ALayer):
    def __init__(self, nb_features: int, nb_neurons: int, activation: str | IActivation) -> None:
        super().__init__(nb_features, nb_neurons, activation)

    def _net_sum(self, X: np.ndarray) -> np.ndarray:
        return (np.dot(X, self.weights) + self.biases)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return (self.activation_function(self._net_sum(X)))
