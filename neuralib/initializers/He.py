##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# He
##

import numpy as np
from typing import Tuple
from .Normal import Normal
from .Uniform import Uniform


class HeNormal(Normal):
    """ Initialize by the value drawn independently from Gaussian distribution whose mean is 0, and standard deviation is set after the He formula."""

    def __init__(self, scale: float = 1.0, use_fan_in=True, rng: np.random.Generator = np.random.default_rng()) -> None:
        super().__init__(scale=scale, rng=rng)
        self.use_fan_in = use_fan_in

    def initialize(self, *, fan_in: int = None, fan_out: int = 1, shape: Tuple[int]) -> np.ndarray:
        if (not fan_in and self.use_fan_in):
            raise "Error: fan_in not provided"
        self.scale *= np.sqrt(2 / (fan_in if self.use_fan_in else fan_out))
        return (super().initialize(shape=shape))


class HeUniform(Uniform):
    """Initialize by the value drawn independently from uniform distribution. low and high are set after the He Formula"""

    def __init__(self, scale: float = 1.0, rng: np.random.Generator = np.random.default_rng()) -> None:
        super().__init__(scale=scale, rng=rng)

    def initialize(self, *, fan_in: int = None, fan_out: int = 1, shape: Tuple[int]) -> np.ndarray:
        if (not fan_in):
            raise "Error: fan_in not provided"
        self.scale *= np.sqrt(6 / fan_in)
        return (super().initialize(shape=shape))
