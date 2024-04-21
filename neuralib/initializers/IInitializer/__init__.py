##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# __init__
##

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Tuple


class IInitializer(ABC):
    """Interface for weight and bias initializers."""

    def __init__(self) -> None:
        super().__init__()
        # np.random.seed(int(time.time()))

    @abstractmethod
    def initialize(self, *, fan_in: int = None, fan_out: int = 1, shape: Tuple[int]) -> np.ndarray:
        """
        Initializes weights or biases with a specific shape.

        Args:
          shape: A tuple representing the shape of the weights/biases to initialize.

        Returns:
          A NumPy array of the initialized weights/biases.
        """
        pass
