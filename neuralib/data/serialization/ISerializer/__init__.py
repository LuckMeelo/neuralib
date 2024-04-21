##
# Project 2024
# neuraltest [WSL : Ubuntu-22.04]
# File description:
# __init__
##

from abc import ABC, abstractmethod


class ISerializer(ABC):

    @abstractmethod
    def serialize(self, obj, filepath: str) -> None:
        """Serializes an object into bytes.

        Args:
            obj: The object to serialize.

        Returns:
            The serialized object as bytes.
        """
        pass

    @abstractmethod
    def deserialize(self, filepath: str) -> any:
        """Deserializes bytes back into an object.

        Args:
            data: The serialized data as bytes.

        Returns:
            The deserialized object.
        """
        pass
