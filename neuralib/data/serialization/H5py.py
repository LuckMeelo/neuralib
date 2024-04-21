##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# H5py
##

from .ISerializer import ISerializer


class H5py(ISerializer):
    def serialize(self, obj, filepath: str) -> None:
        pass

    def deserialize(self, filepath: str) -> any:
        pass
