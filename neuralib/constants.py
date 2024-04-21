##
# EPITECH PROJECT, 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# constants
##

# Activation functions ID

from enum import Enum


# TODO add other constants

class ActivationID(str, Enum):
    BINARY_STEP = 'binary_step'
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    CLIPPED_RELU = 'clipped_relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SOFTMAX = 'softmax'


class LossID(str, Enum):
    CROSSENTROPY_CATEGORICAL = 'categorical_cross_e'
    CROSSENTROPY_BINARY = 'binary_cross_e'
    MEAN_ABOSOLUTE_ERROR = 'mae'
    MEAN_SQUARED_ERROR = 'mse'
