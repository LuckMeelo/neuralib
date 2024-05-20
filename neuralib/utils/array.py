##
# Project 2024
# neural_test [WSL : Ubuntu-22.04]
# File description:
# array
##

import numpy as np


# TODO delete
def display_2D_dims(arr: np.ndarray, verbose=False) -> None:
    if (verbose):
        print(arr)
    if (len(arr.shape) not in [1, 2]):
        raise "Shape error"
    x = arr.shape[0] if (len(arr.shape) == 1) else arr.shape[1]
    y = 1 if (len(arr.shape) == 1) else arr.shape[0]
    return (f"{x}x{y}")
