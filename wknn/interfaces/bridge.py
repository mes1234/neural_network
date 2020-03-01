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

    @abc.abstractmethod
    def train(self)->None:
        """
        train child and back propagate error to parent
        """



    @property
    @abc.abstractmethod
    def X(self)->list:
        """
        return input vector of parent
        
        Returns:
            list -- return input vector of parent minus bias
        """ 


    @X.setter
    @abc.abstractmethod
    def X(self,value: list)->None:
        """
        Assign value vector into input of parent
        
        Arguments:
            value {list} -- X vector of parent
        """

    @property
    @abc.abstractmethod
    def ytrain(self,)->list:
        """
        return desired output of child
        
        Returns:
            list -- desired output of child
        """

    @ytrain.setter
    @abc.abstractmethod
    def ytrain(self,value: list)->None:
        """
        Assign value vector into desired output of child
        
        Arguments:
            value {list} -- ytrain vector of child
        """