from wknn.models.units.sigmoid_neuron import Sigmoid_Neuron



class Back_Prop_Neuron(Sigmoid_Neuron):

    def __init__(self):
        super().__init__()
        self.delta = 0.05
        self._learning_rate = 20.2
    
    
    def _calculate_dout(self,index):
        weights = self.W.copy()
        self.W[index] = self.W_delta[index]
        diff = self.out -self.cached_out
        self.W = weights
        return diff

    def _adjust_W(self,w_d):
        return w_d[0]+self.error*w_d[1]*self._learning_rate

    @property
    def back_prop_error(self)->list:
        '''
        return back propagated list of error
        '''
        d = list(self.d)
        self._sum_d = sum(d) 
        return   map(lambda x:x/self._sum_d*self.error,d) 
    @property
    def d(self)->list:
        '''
        derivative of error over inputs
        '''
        self.W_delta = list((x+self.delta for x in self.W))
        self.cached_out = self.out
        self.out_delta = map(self._calculate_dout, range(len(self.X)))
        return (x/self.delta for x in self.out_delta)
    
    def train(self):
        self.W = list(map(self._adjust_W,zip(self.W,self.d)))
