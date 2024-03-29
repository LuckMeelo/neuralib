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

# load csv data
X, y = ml.data.load_csv("./data/AND_dataset.csv", "output")

# perceptron initialization
perceptron = ml.Perceptron(n_features=2, activation='binary_step')

# perceptron training
perceptron.train(X_sample=X, y_expected=y)

# perceptron prediction on test data sets
predicted = perceptron.predict(X)

# output predictions
print(predicted)

# test prediction value
print((predicted == np.reshape(y, -1)).all())

# save state to a json file
perceptron.save_to_json("perceptron.json")

# load the json file
pp = ml.Perceptron.from_json("perceptron.json")

# perceptron prediction on test data sets
predicted = pp.predict(X)

# output predictions
print(predicted)

# test prediction value
print((predicted == np.reshape(y, -1)).all())
