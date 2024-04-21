##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# One
##

import numpy as np
from typing import Tuple
from .IInitializer import IInitializer


class One(IInitializer):
    """Initializes array full of ones."""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self, *, fan_in: int = None, fan_out: int = 1, shape: Tuple[int]) -> np.ndarray:
        return (np.ones(shape))
