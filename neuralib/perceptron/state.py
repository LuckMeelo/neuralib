##
# LuckMeelo, 2023
# neuralib
# File description:
# perceptron_state
##

import numpy as np
from neuralib.initializers import InitializerInterface, Zeros
from typing import Callable


class PerceptronState:
    """
        A Perceptrion State class storing infos:
            - nb_inputs
            - learning rate (0.01 by default)
            - weights (nb_inputs long)
            - bias
            - activation function
    """

    def __init__(self, nb_inputs: int, activation: Callable[[float], float], out: int = 1,
                 learning_rate: float = 0.01, initializer: InitializerInterface = Zeros()) -> None:
        """
            Initialize a new perceptron state
        """
        self.nb_inputs = nb_inputs
        self.learning_rate = learning_rate
        self.weights = initializer.init_weights(fan_in=nb_inputs, fan_out=out)
        self.bias = initializer.init_bias(fan_in=nb_inputs, fan_out=out)
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
