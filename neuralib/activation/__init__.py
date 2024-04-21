##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# __init__
##

from neuralib.constants import ActivationID

from .IActivation import IActivation
from .BinaryStep import BinaryStep
from .ReLU import ReLU, LeakyReLU, ClippedReLU
from .Sigmoid import Sigmoid
from .Tanh import Tanh
from .Softmax import Softmax


default_activation_functions = {
    ActivationID.BINARY_STEP: BinaryStep,
    ActivationID.RELU: ReLU,
    ActivationID.LEAKY_RELU: LeakyReLU,
    ActivationID.CLIPPED_RELU: ClippedReLU,
    ActivationID.SIGMOID: Sigmoid,
    ActivationID.TANH: Tanh,
    ActivationID.SOFTMAX: Softmax,
}


def getActivationFromID(id: str) -> IActivation:
    id = id.lower()
    if (id not in default_activation_functions):
        raise "Error activation function " + id + " not found"
    return (default_activation_functions[id]())
