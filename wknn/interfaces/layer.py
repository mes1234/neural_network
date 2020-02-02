import abc


class Layer(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def Items(self)->list:
        return
    
    @Items.setter
    @abc.abstractmethod
    def Items(self, new_value) -> None:
        return

    @abc.abstractmethod
    def run()-> None:
        return

    @Input.setter
    @abc.abstractmethod
    def Input(self,input_vector:list)-> None:
        return

    @property
    @abc.abstractmethod
    def Output(self)->list:
        return

    @abc.abstractmethod
    def connect(self,previous:Layer,next:Layer)->None
        return

    @abc.abstractmethod
    def train(self) -> None:

