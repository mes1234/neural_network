import abc
from wknn.interfaces.layer import Layer


class Bridge(metaclass = abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, parent : Layer, child : Layer )->None:
        pass

    @abc.abstractmethod
    def forward(self)->None:
        """
        execute parent and pass its result to child as input
        """
        pass

    @abc.abstractmethod
    def train(self)->None:
        """
        train child and back propagate error to parent
        """
        pass