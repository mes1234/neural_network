from wknn.models.units.perceptron import Perceptron
from wknn.utilities.output_fuctions import sigmoid_funct

import math

class Sigmoid_Neuron(Perceptron):

    def __init__(self):
        super().__init__()
        self.output_func = sigmoid_funct
