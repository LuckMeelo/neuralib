##
# LuckMeelo, 2023
# neuralib
# File description:
# perceptron_state
##

import numpy as np
from neuralib.activation import activation_functions
from neuralib.initializers import InitializerInterface, Zeros


class PerceptronState:
    """
        A Perceptrion State class storing infos:
            - nb_inputs
            - learning rate (0.01 by default)
            - weights (nb_inputs long)
            - bias
            - activation function
    """

    def __init__(self, nb_inputs: int, activation: str,
                 learning_rate: float = 0.01, initializer: InitializerInterface = Zeros()) -> None:
        """
            Initialize a new perceptron state
        """
        self.nb_inputs = nb_inputs
        self.learning_rate = learning_rate
        self.weights = initializer.init_weights(shape=nb_inputs)
        self.bias = initializer.init_bias()
        activation = activation.lower()
        if (activation not in activation_functions):
            raise "error invalid activation function"
        self.activation = activation

    def update_weights(self, weights: np.ndarray) -> None:
        """
            Set new weights
        """
        self.weights = weights

    def update_bias(self, bias: float) -> None:
        """
            Set new bias
        """
        self.bias = bias
