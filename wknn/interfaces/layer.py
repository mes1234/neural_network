import abc
from wknn.models.units.back_prop_neuron import Back_Prop_Neuron


class Layer(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def __init__(self,*,inputs_count, neuron_count  ,neuron_type = Back_Prop_Neuron  )->list:
        return
   

    @property
    @abc.abstractmethod
    def X(self)->list:
        return

    @X.setter
    @abc.abstractmethod
    def X(self,value:list)->None:
        return
