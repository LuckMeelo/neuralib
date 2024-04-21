##
# EPITECH PROJECT, 2024
# neural_test [WSL : Ubuntu-22.04]
# File description:
# Denselayer
##

from .ILayer import ALayer
import numpy as np
from neuralib.activation import IActivation
# from initializers import IInitializer, Uniform, Zeros


class DenseLayer(ALayer):
    def __init__(self, nb_features: int, nb_neurons: int, activation: str | IActivation) -> None:
        super().__init__(nb_features, nb_neurons, activation)

    def _net_sum(self, X: np.ndarray) -> np.ndarray:
        return (np.dot(X, self.weights) + self.biases)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return (self.activation_function(self._net_sum(X)))
