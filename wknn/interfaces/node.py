import abc

class Node(metaclass=abc.ABCMeta):


    @property
    @abc.abstractmethod
    def X(self)->list:
        return
    @X.setter
    @abc.abstractmethod
    def X(self,new_value)->None:
        return

    @property
    @abc.abstractmethod
    def W(self)->list:
        return
    @W.setter
    @abc.abstractmethod
    def W(self,new_value)->None:
        return

    @property
    @abc.abstractmethod
    def y(self)->float:
        return

    @property
    @abc.abstractmethod
    def ytrain(self)->float:
        return
    @ytrain.setter
    @abc.abstractmethod
    def ytrain(self)->None:
        return
    @property
    @abc.abstractmethod
    def error(self)->None:
        return

    @abc.abstractmethod
    def run(self):
        return
    @abc.abstractmethod
    def train(self):
        return