from wknn.interfaces.bridge import Bridge
from wknn.interfaces.layer import Layer


class Simple_Bridge(Bridge):
    def __init__(self, parent : Layer, child : Layer )->None:
        self.parent = parent
        self.child = child
    

    def forward(self)->None:
        """
        execute parent and pass its result to child as input
        """
        pass

    def train(self)->None:
        """
        train child and back propagate error to parent
        """
        pass