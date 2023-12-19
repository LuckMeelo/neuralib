##
# EPITECH PROJECT, 2023
# neuralib [WSL : Ubuntu-22.04]
# File description:
# zero
##

from typing import Tuple
import numpy as np
from neuralib.initializers.initializer_interface import InitializerInterface


class Zeros(InitializerInterface):
    """
        Zero initializer for weights and bias
    """

    def __init__(self) -> None:
        super().__init__()

    def init_weights(self, *, shape: Tuple[int] | int, fan_out: int = 0) -> np.ndarray:
        return (np.zeros(shape))
