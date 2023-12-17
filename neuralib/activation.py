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

#  Callable[[np.ndarray], np.ndarray]


def binary_step(y: np.ndarray) -> np.ndarray:
    # 1 if y >= 0 else 0
    return (np.where(y >= 0, 1, 0))


activation_functions = {
    'binary_step': binary_step,
}
