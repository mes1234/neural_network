import abc

from wknn.interfaces.layer import Layer
from wknn.models.units.back_prop_neuron import Back_Prop_Neuron

from random import random

class Simple_Layer(Layer):

    def __init__(self,*,inputs_count, neuron_count  ,neuron_type = Back_Prop_Neuron  )->list:
        
        self._items = [neuron_type() for x in range(neuron_count)]
        for item in self._items:
            item.W = [random() for x in range(inputs_count)]

    @property
    def X(self)->list:
        return self._X

    @X.setter
    def X(self,value:list)->None:
        list(map(lambda x:setattr(x,"X",value),  self._items))