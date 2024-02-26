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
    return (np.where(y >= 0, 1, 0))


def tanh(y: np.ndarray, *, derivative=False) -> np.ndarray:
    return np.tanh(y)


def relu(y: np.ndarray, *, derivative=False) -> np.ndarray:
    return np.maximum(0, y)


def softmax(y: np.ndarray, *, derivative=False) -> np.ndarray:
    exp_scores = np.exp(y)
    return exp_scores / np.sum(exp_scores)


def sigmoid_function(y: np.ndarray, *, derivative=False) -> np.ndarray:
    return (1/(1 + np.exp(- y)))


# TODO also store derivatives

activation_functions = {
    'binary_step': binary_step,
    'tanh': tanh,
    'relu': relu,
    'softmax': softmax,
}
