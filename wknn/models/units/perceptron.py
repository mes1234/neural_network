from wknn.models.units.dummy import Dummy
from functools import reduce
from operator import add, sub, mul

class Perceptron(Dummy):

    def __init__(self):
        super().__init__()
        self._learning_rate = 1.0

    def run(self):
        zipped = (zip(self.X,self.W))
        self._y = (x*w for x,w in (zipped))
        pass
    def train(self):
        self._error = self.ytrain - self.y
        if self._error >=0.0:
            operation = sub
        else:
            operation = sum
        X = (self._learning_rate*x for x in self.X)
        self.W = list(map(operation,zip(self.W,X)))
        pass

