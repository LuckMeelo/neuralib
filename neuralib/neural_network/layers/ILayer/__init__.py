##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# __init__
##

import numpy as np
from abc import ABC, abstractmethod
from neuralib.initializers import IInitializer, Uniform, Zeros
from neuralib.activation import IActivation, getActivationFromID


class ILayer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self,  fan_out: int = 1, weights_init: IInitializer = Uniform(),
                   biases_init: IInitializer = Zeros()) -> None:
        pass

    @abstractmethod
    def set_weights(self, w: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_biases(self, b: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_biases(self) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Performs the forward pass of the layer.

        Args:
            X: The input data, typically a NumPy array.

        Returns:
            A NumPy array representing the layer's output.
        """

class ALayer(ILayer):
    # TODO: Maybe add identity function
    def __init__(self, nb_features: int, nb_neurons: int, activation: str | IActivation) -> None:
        self.activation_function = getActivationFromID(
            activation) if isinstance(activation, str) else activation
        self.nb_features = nb_features
        self.nb_neurons = nb_neurons
        self.weights = np.empty(None)
        self.biases = np.empty(None)

    def __str__(self) -> str:
        return f"""Layer state:
                - nb_features => {self.nb_features}
                - nb_neurons => {self.nb_neurons}
                - weights => {self.weights}
                - biases => {self.biases}
                - activation function => {self.activation_function.get_id()}
                """

    def initialize(self,  fan_out: int = 1, weights_init: IInitializer = Uniform(),
                   biases_init: IInitializer = Zeros()) -> None:
        fan_in = self.nb_features
        weights_shape = (self.nb_features, self.nb_neurons)
        biases_shape = (1, self.nb_neurons)

        self.weights = weights_init.initialize(fan_in=fan_in, fan_out=fan_out,
                                               shape=weights_shape)
        self.biases = biases_init.initialize(fan_in=fan_in, fan_out=fan_out,
                                             shape=biases_shape)

    def set_weights(self, w: np.ndarray) -> None:
        self.weights = w

    def set_biases(self, b: np.ndarray) -> None:
        self.biases = b

    def get_weights(self) -> np.ndarray:
        return (self.weights)

    def get_biases(self) -> np.ndarray:
        return (self.biases)
