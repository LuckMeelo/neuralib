##
# Project 2024
# neuralib [WSL : Ubuntu-22.04]
# File description:
# neural_sample
##

import sys
import numpy as np
import matplotlib
from typing import List

import neuralib.data.generator as gen
from neuralib.neural_network import NeuralNetwork
from neuralib.activation import BinaryStep
from neuralib.loss import BinaryCrossEntropy
from neuralib.neural_network import DenseLayer
from neuralib.utils import display_2D_dims

import pandas
print("Python:", sys.version)
print("Numpy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pandas.__version__)
print()


def print_infos_and_forward(X: np.ndarray, l: DenseLayer) -> np.ndarray:
    print("intputs => ", display_2D_dims(X, verbose=True))
    print("weights => ", display_2D_dims(l.weights, verbose=True))
    print("biases => ", display_2D_dims(l.biases, verbose=True))
    output = l.forward(X)
    return (output)


# def test_dense_layer() -> int:
#     # X = np.array([1, 2, 3, 2.5])
#     X, y = gen.spiral_data(2, 2)
#     layer1 = DenseLayer(nb_features=2, nb_neurons=5, activation='relu')
#     layer1.initialize(fan_out=3)
#     layer2 = DenseLayer(nb_features=5, nb_neurons=3, activation='relu')
#     layer2.initialize()

#     layer1_output = print_infos_and_forward(X, layer1)
#     print("Result1: ", layer1_output)
#     print("=" * 30)
#     print("Result2: ", print_infos_and_forward(layer1_output, layer2))
#     return (0)


def test_neural_network() -> int:
    # X = np.array([1, 2, 3, 2.5])
    X, y = gen.spiral_data(2, 2)
    nn = NeuralNetwork(nb_features=2)
    nn.addLayer(layer=DenseLayer(nb_features=2,
                nb_neurons=5, activation='relu'))
    nn.addLayer(layer=DenseLayer(nb_features=5,
                nb_neurons=3, activation='relu'))

    nn.compile(loss=BinaryCrossEntropy())
    output = nn.predict(X)

    print("Result: ", output)
    return (0)


def main(ac: int, av: List[str]) -> int:
    # test_dense_layer()
    test_neural_network()
    return (0)


if __name__ == "__main__":
    av = sys.argv
    ac = len(av)
    ret = main(ac, av)
    sys.exit(ret)
