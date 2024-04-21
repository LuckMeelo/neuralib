##
# Project 2024
# neural_test [WSL : Ubuntu-22.04]
# File description:
# Constant
##

import numpy as np
from typing import Tuple
from .IInitializer import IInitializer


class Zeros(IInitializer):
    """Initializes array full of zeros."""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self, *, fan_in: int = None, fan_out: int = 1, shape: Tuple[int]) -> np.ndarray:
        return (np.zeros(shape))
