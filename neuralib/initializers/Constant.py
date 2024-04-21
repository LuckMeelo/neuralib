##
# EPITECH PROJECT, 2024
# neural_test [WSL : Ubuntu-22.04]
# File description:
# Constant
##

from .IInitializer import IInitializer
from typing import Tuple
import numpy as np


class Constant(IInitializer):
    """Initializes array with constant value."""

    def __init__(self, fill_value: float) -> None:
        super().__init__()
        self.fill_value = fill_value

    def initialize(self, *, fan_in: int = None, fan_out: int = 1, shape: Tuple[int]) -> np.ndarray:
        array = np.empty(shape)
        array.fill(self.fill_value)
        return (array)
