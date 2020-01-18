import abc

class Node(object):
    ___metaclass__ = abc.ABCMeta

    # X vector
    def X_getter(self)->list:
        return []
    def X_setter(self,new_value)->None:
        return
    X = abc.abstractproperty(fget=X_getter,fset=X_setter,doc="Input vector for Node")
    #
    # W vector
    def W_getter(self)->list:
        return []
    def W_setter(self,new_value)->None:
        return
    W = abc.abstractproperty(fget=W_getter,fset=W_setter,doc="Input weight vector for Node")
    #
    # Y 
    def y_getter(self)->float:
        return []
    y = abc.abstractproperty(fget=y_getter,fset=None,doc="Output value of Node")
    #
    # Y 
    def ytrain_getter(self)->float:
        return []
    ytrain = abc.abstractproperty(fget=ytrain_getter,fset=None,doc="Intended output value of Node")
    #
    @abc.abstractmethod
    def run(self):
        return
    @abc.abstractmethod
    def train(self):
        return