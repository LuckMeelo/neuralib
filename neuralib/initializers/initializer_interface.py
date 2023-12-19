##
# Luck Meelo, 2023
# neuralib
# File description:
# initializer_interface
##

from typing import Tuple
import numpy as np
import time


class InitializerInterface:
    """
        Neurons data initializers interface
    """

    def __init__(self) -> None:
        np.random.seed(int(time.time()))

    def init_weights(self, *, shape: Tuple[int] | int, fan_out: int = 0) -> np.ndarray:
        """
            Generates weights for neurons in the neural network
        """
        raise NotImplementedError()

    def init_bias(self) -> float:
        return (0.1)
