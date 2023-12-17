##
# luckMeelo, 2023
# neuralib [WSL : Ubuntu-22.04]
# File description:
# perceptron
##

import numpy as np
from typing import Callable
from neuralib.activation import activation_functions
from neuralib.initializers import InitializerInterface, Zeros
from neuralib.perceptron.state import PerceptronState


class Perceptron():
    def __init__(self, nb_inputs: int, activation: str, nb_out: int = 1,
                 learning_rate: float = 0.01, initializer: InitializerInterface = Zeros()) -> None:
        self.state = PerceptronState(nb_inputs=nb_inputs, activation=activation,
                                     nb_out=nb_out, learning_rate=learning_rate, initializer=initializer)

    @classmethod
    def from_state(cls, state: PerceptronState):
        p = cls(nb_inputs=state.nb_inputs, activation=state.activation)
        p.load_state(state)
        return ()

    def load_state(self, state: PerceptronState) -> None:
        self.state = state

    def get_state(self) -> PerceptronState:
        return (self.state)

    def fit(self, X_sample: np.ndarray, y_expected: np.ndarray, epochs: int = 100):
        for _ in range(epochs):
            for idx, X in enumerate(X_sample):
                y_predicted = self.predict(X)
                error = y_expected[idx] - y_predicted
                self.update_weights(error, X)
                self.update_bias(error)

    def update_weights(self, error: np.ndarray, X_sample: np.ndarray) -> None:
        self.state.weights += self.state.learning_rate * error * X_sample

    def update_bias(self, error: np.ndarray) -> None:
        self.state.bias += self.state.learning_rate * error

    def predict(self, X_sample: np.ndarray) -> np.ndarray:
        return activation_functions[self.state.activation](np.dot(X_sample, self.state.weights) + self.state.bias)
