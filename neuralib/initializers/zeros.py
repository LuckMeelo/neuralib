##
# EPITECH PROJECT, 2023
# neuralib [WSL : Ubuntu-22.04]
# File description:
# zero
##

import numpy as np
from initializer_interface import InitializerInterface


class Zeros(InitializerInterface):
    """
        Zero initializer for weights and bias
    """

    def __init__(self) -> None:
        super().__init__()

    def init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        return (np.zeros(fan_in))

    def init_bias(self, fan_in: int, fan_out: int) -> float:
        return (0.0)
