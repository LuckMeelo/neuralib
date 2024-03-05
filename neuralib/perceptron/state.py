##
# LuckMeelo, 2023
# neuralib
# File description:
# perceptron_state
##

import json
import numpy as np
from neuralib.activation import activation_functions
from neuralib.initializers import InitializerInterface, Zeros


class PerceptronState:
    """
        A Perceptrion State class storing infos:
            - n_features
            - learning rate (0.01 by default)
            - weights (n_features long)
            - bias
            - activation function
    """

    def __init__(self, n_features: int, activation: str,
                 learning_rate: float = 0.01, initializer: InitializerInterface = Zeros()) -> None:
        """
            Initialize a new perceptron state
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.weights = initializer.init_weights(shape=n_features)
        self.bias = initializer.init_bias()
        activation = activation.lower()
        if (activation not in activation_functions):
            raise "error invalid activation function"
        self.activation = activation

    @classmethod
    def from_json(cls, filepath: str) -> None:
        # TODO
        with open(filepath, "r") as f:
            data = json.load(f)
        st = cls(n_features=data["n_features"], activation=data["activation"],
                 learning_rate=float(data["learning_rate"]))
        st.set_weights(np.array(data["weights"]))
        st.set_bias(np.array(data["bias"]))
        return (st)

    def save_to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.get_data(), f, indent=4)

    def get_data(self) -> dict:
        # Convert NumPy array to list
        weights_list = self.weights.tolist()
        bias_list = self.bias.tolist()

        data = {
            "n_features": self.n_features,
            "weights": weights_list,
            "bias": bias_list,
            "learning_rate": self.learning_rate,
            "activation": self.activation
        }
        return (data)

    def set_weights(self, weights: np.ndarray) -> None:
        """
            Set new weights
        """
        self.weights = weights

    def set_bias(self, bias: float) -> None:
        """
            Set new bias
        """
        self.bias = bias
