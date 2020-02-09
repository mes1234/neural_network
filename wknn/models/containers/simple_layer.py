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
        """return input vector for Layer
        
        Returns:
            list -- vector of input values
        """
        res = list(map(lambda x:getattr(x,"X"),  self._items))
        return res

    @X.setter
    def X(self,value:list)->None:
        """set input for every Neuron in layer
        
        Arguments:
            value {list} -- list which will be applied to every item in layer
        """
        list(map(lambda x:setattr(x,"X",value),  self._items))

    @property
    def ytrain(self)->list:
        """fetch currently set value of desired output, apply only to output layer
        
        Returns:
            list -- [description]
        """
        res = list(map(lambda x:getattr(x,"ytrain"),  self._items))
        return res

    @ytrain.setter
    def ytrain(self,value:list)->None:
        """set desired output value for every neuron in layer
        
        Arguments:
            value {list} -- list which will be applied to items in layer
        """
        list(map(lambda x:setattr(x[0],"ytrain",x[1]),  zip(self._items,value)))