from wknn.models.units.dummy import Dummy
from wknn.utilities.output_fuctions import step_funct
from functools import reduce
from operator import add, sub, mul

import math

class Perceptron(Dummy):

    def __init__(self):
        super().__init__()
        self._learning_rate = 0.7
        self.output_func = step_funct

    def run(self):
        zipped = (zip(self.X,self.W))
        self._y = (x*w for x,w in (zipped))
        pass
    def train(self):
        X = (self._learning_rate*x for x in self.X)
        self.W = list(map(lambda x:x[0]+self.error*x[1],zip(self.W,X)))
        pass
    
    @property
    def out(self):
        return self.output_func(self.y)

    @property
    def error(self):
        return self.ytrain - self.out