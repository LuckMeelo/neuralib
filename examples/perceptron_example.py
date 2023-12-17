##
# EPITECH PROJECT, 2023
# neuralib [WSL : Ubuntu-22.04]
# File description:
# perceptron_example
##

# construct
# add
# compile
# fit/ train
# evaluate (stats -> loss)
# predict on tests

import numpy as np
import neuralib as ml

X, y = ml.data.load_csv("./data/AND_dataset.csv", "output")
perceptron = ml.Perceptron(nb_inputs=2, activation='binary_step')
perceptron.fit(X_sample=X, y_expected=y)
predicted = perceptron.predict(X)
print(predicted)
print((predicted == np. reshape(y, -1)).all())
