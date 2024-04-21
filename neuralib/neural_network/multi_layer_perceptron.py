##
# EPITECH PROJECT, 2024
# neuralib [WSL: Ubuntu-22.04]
# File description:
# cnn
##


from .layers import ILayer
from neuralib.loss import ILoss, getLossFromID
from neuralib.initializers import IInitializer, Uniform, Zeros
import numpy as np
from neuralib.data.serialization.Pickle import Pickle


class NeuralNetwork():
    def __init__(self, nb_features: int) -> None:
        self.nb_features = nb_features
        self.layers = []
        self.loss_function = None
        self.compiled = False

    def addLayer(self, layer: ILayer, weights_init: IInitializer = Uniform(),
                 biases_init: IInitializer = Zeros()) -> None:
        self.layers.append((layer, weights_init, biases_init))

    def _check_compiled(self):
        if (not self.compiled):
            raise "Error network not compiled"

    def _compile_layers(self) -> None:
        if (not self.layers):
            raise "No layers added"
        nb_layers = len(self.layers)
        for i in range(nb_layers - 1):
            layer, wi, bi = self.layers[i]
            layer.initialize(
                fan_out=self.layers[i+1][0].nb_neurons, weights_init=wi, biases_init=bi)
            self.layers[i] = layer
        layer, wi, bi = self.layers[-1]
        layer.initialize(weights_init=wi, biases_init=bi)
        self.layers[-1] = layer
        self.compiled = True

    def compile(self, loss: str | ILoss) -> None:
        self.loss_function = getLossFromID(
            loss) if isinstance(loss, str) else loss
        self._compile_layers()

    def train(self, X: np.ndarray, y: np.ndarray, epochs=1000) -> np.ndarray:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        inputs = X
        for layer in self.layers:
            outputs = layer.forward(inputs)
            inputs = outputs
        return (outputs)

    def evaluate(self):
        pass

    def save(self, filepath: str) -> None:
        self._check_compiled()
        serializer = Pickle()
        serializer.serialize(self, filepath)

    def load(self, filepath: str) -> None:
        serializer = Pickle()
        self = serializer.deserialize(filepath)

    @classmethod
    def fromModel(cls, filepath: str):
        serializer = Pickle()
        nn = serializer.deserialize(filepath)
        return (nn)
