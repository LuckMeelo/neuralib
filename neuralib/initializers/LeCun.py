##
# EPITECH PROJECT, 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# LeCun
##

from .Uniform import Uniform
from .Normal import Normal
from typing import Tuple
import numpy as np


class LeCunNormal(Normal):
    """ Initialize by the value drawn independently from Gaussian distribution whose mean is 0, and standard deviation is set after leCun Formula"""

    def __init__(self, scale: float = 1.0, rng: np.random.Generator = np.random.default_rng()) -> None:
        super().__init__(scale=scale, rng=rng)

    def initialize(self, *, fan_in: int = None, fan_out: int = 1, shape: Tuple[int]) -> np.ndarray:
        if (not fan_in):
            raise "Error: fan_in not provided"
        self.scale *= np.sqrt(1 / fan_in)
        return (super().initialize(shape=shape))


class LeCunUniform(Uniform):
    """Initialize by the value drawn independently from uniform distribution. low and high are set after the leCun Formula"""

    def __init__(self, scale: float = 1.0, rng: np.random.Generator = np.random.default_rng()) -> None:
        super().__init__(scale=scale, rng=rng)

    def initialize(self, *, fan_in: int = None, fan_out: int = 1, shape: Tuple[int]) -> np.ndarray:
        if (not fan_in):
            raise "Error: fan_in not provided"
        self.scale *= np.sqrt(3 / fan_in)
        return (super().initialize(shape=shape))
