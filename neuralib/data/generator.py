##
# EPITECH PROJECT, 2024
# neural_test [WSL : Ubuntu-22.04]
# File description:
# generation
##

import numpy as np
from typing import Tuple

# https://cs231n.github.io/neural-networks-case-study/


def spiral_data(points: int, classes: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y