from wknn.interfaces.bridge import Bridge
from wknn.interfaces.layer import Layer


class Simple_Bridge(Bridge):
    def __init__(self, parent: Layer, child: Layer) -> None:
        self.parent = parent
        self.child = child
        self.out = []

    def forward(self) -> None:
        """
        execute parent and pass its result to child as input
        """
        self.child.X =[1.] + self.parent.out
        self.out = self.child.out

    def train(self) -> None:
        """
        train child and back propagate error to parent
        """
        self.forward()
        bpe = self.child.back_prop_error
        pout = self.parent.out
        self.parent.ytrain = list(map(lambda x: x[0]+x[1],zip(pout,bpe[1:])))
        self.child.train()
        self.parent.train()

    @property
    def X(self)->list:
        """
        return input vector of parent
        
        Returns:
            list -- return input vector of parent minus bias
        """
        return self.parent.X[0][1:]



    @X.setter
    def X(self, value: list) -> None:
        """
        Assign value vector into input of parent

        Arguments:
            value {list} -- X vector of parent
        """
        self.parent.X = [1.0]+value

    @property
    def ytrain(self,)->list:
        """
        return desired output of child
        
        Returns:
            list -- desired output of child
        """
        return self.child.ytrain


    @ytrain.setter
    def ytrain(self, value: list) -> None:
        """
        Assign value vector into desired output of child

        Arguments:
            value {list} -- ytrain vector of child
        """
        self.child.ytrain = value
