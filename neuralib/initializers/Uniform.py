##
# EPITECH PROJECT, 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# Normal
##

from .IInitializer import IInitializer
from typing import Tuple
import numpy as np


class Uniform(IInitializer):
    """Initialize by the value drawn independently from uniform distribution."""

    def __init__(self, scale: float = 0.05, rng: np.random.Generator = np.random.default_rng()) -> None:
        super().__init__()
        self.scale = scale
        self.rng = rng

    def initialize(self, *, fan_in: int = None, fan_out: int = 1, shape: Tuple[int]) -> np.ndarray:
        return (self.rng.uniform(low=-self.scale, high=self.scale, size=shape))
