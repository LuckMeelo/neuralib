##
# EPITECH PROJECT, 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# serialization
##

from .ISerializer import ISerializer
import pickle


class Pickle(ISerializer):
    def serialize(self, obj, filepath: str) -> None:
        with open(filepath, "wb") as output_file:
            pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)

    def deserialize(self, filepath: str) -> any:
        with open(filepath, "rb") as input_file:
            obj = pickle.load(input_file)
            return (obj)
