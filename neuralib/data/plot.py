##
# Project 2024
# neural_test [WSL : Ubuntu-22.04]
# File description:
# plotting
##

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_data(X: np.ndarray, y: np.ndarray) -> None:
    matplotlib.use('TkAgg')

    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
