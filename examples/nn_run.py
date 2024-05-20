##
# Project 2024
# neuralib [WSL : Ubuntu-22.04]
# File description:
# neural_sample
##

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List

from neuralib.data.dataset_io import load_csv
from neuralib.neural_network import NeuralNetwork, DenseLayer
import pandas

print("Python:", sys.version)
print("Numpy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pandas.__version__)
print()


def mnist_random_plot(X_train: np.ndarray, y_train: np.ndarray):
    matplotlib.use('TkAgg')
    fig, axes = plt.subplots(2,5, figsize=(12,5))
    axes = axes.flatten()
    idx = np.random.randint(0,42000,size=10)
    for i in range(10):
        axes[i].imshow(X_train[idx[i],:].reshape(28,28), cmap='gray')
        axes[i].axis('off') # hide the axes ticks
        axes[i].set_title(str(int(y_train[idx[i]])), color= 'black', fontsize=25)
    plt.show()

def main(ac: int, av: List[str]) -> int:
    X_train, y_train = load_csv(file_path="./samples/train.csv", target_columns=["label"])
    X_test = load_csv(file_path="./samples/test.csv")
    
    nn = NeuralNetwork(nb_features=784)
    
    nn.addLayer(DenseLayer(nb_features=784, nb_neurons=256, activation="relu"))
    nn.addLayer(DenseLayer(nb_features=256, nb_neurons=10, activation="softmax"))
    nn.compile(loss="categorical_cross_e")
    output = nn.predict(X_test)
    print(output)
    return (0)




if __name__ == "__main__":
    av = sys.argv
    ac = len(av)
    ret = main(ac, av)
    sys.exit(ret)
