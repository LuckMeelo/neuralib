##
# LuckMeelo, 2023
# neuralib
# File description:
# activation
##

import numpy as np

# Binary Step
# Sigmoid or Logistic Activation Function aka Soft Step
# Tanh or hyperbolic tangent Activation Function
# ArcTan activation function
# ReLU (Rectified Linear Unit) Activation Function
# soft max
# sigmoid function or Logistic activation function

#  Callable[[np.ndarray], np.ndarray]


def binary_step(y: np.ndarray, *, derivative=False) -> np.ndarray:
    if (derivative):
        return (0)
    return (np.where(y >= 0, 1, 0))


def tanh(y: np.ndarray, *, derivative=False) -> np.ndarray:
    if (derivative):
        return (1 - (np.tanh(y) ** 2))
    return np.tanh(y)


def relu(y: np.ndarray, *, derivative=False) -> np.ndarray:
    if (derivative):
        return np.where(y >= 0, 1, 0)
    return np.maximum(0, y)


def leaky_relu(y: np.ndarray, *, derivative=False) -> np.ndarray:
    if (derivative):
        return np.where(y >= 0, 1, 0.01)
    return np.maximum(0.1*y, y)


def sigmoid_function(y: np.ndarray, *, derivative=False) -> np.ndarray:
    if (derivative):
        sig = sigmoid_function(y, derivative=False)
        return (sig * (1 - sig))
    return (1/(1 + np.exp(- y)))


def softmax(y: np.ndarray, *, derivative=False) -> np.ndarray:
    exp_scores = np.exp(y)
    return exp_scores / np.sum(exp_scores)

# TODO also store derivatives


activation_functions = {
    'binary_step': binary_step,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'softmax': softmax,
}
