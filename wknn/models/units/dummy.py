import abc

from wknn.interfaces.node import Node

class Dummy(Node):
    __doc__ = "Implements monst common elements of units, should not be implemented"

    def __init__(self):
        self._X = []
        self._W = []
        self._y = None
        self._ytrain = None
        
    @property
    def X(self)->list:
        return self._X
    @X.setter
    def X(self,new_value)->None:
        self._X = new_value
    #
    @property
    def W(self)->list:
        return self._W
    @W.setter
    def W(self,new_value)->None:
        self._W = new_value
    #
    @property 
    def y(self)->float:
        self.run()
        if self._y is not None:
            return sum(self._y)
        else:
            raise ValueError("X and W not set")
    #
    @property
    def ytrain(self)->float:
        return self._ytrain
    @ytrain.setter
    def ytrain(self,new_value)->float:
        self._ytrain = new_value


Node.register(Dummy)