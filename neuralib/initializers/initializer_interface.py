##
# Luck Meelo, 2023
# neuralib
# File description:
# initializer_interface
##

import numpy as np
import time


class InitializerInterface:
    """
        Neurons data initializers interface
    """

    def __init__(self) -> None:
        np.random.seed(int(time.time()))

    def init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        """
            Generates weights for neurons in the neural network
        """
        raise NotImplementedError()

    def init_bias(self, fan_in: int, fan_out: int) -> float:
        """
            Generates bias value for neurons in the neural network
        """
        raise NotImplementedError()
