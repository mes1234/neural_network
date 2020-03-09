from wknn.models.containers.simple_layer import Simple_Layer
from wknn.models.containers.simple_bridge import Simple_Bridge


def test_init():
    l1 = Simple_Layer(inputs_count=2, neuron_count=10)
    l2 = Simple_Layer(inputs_count=10, neuron_count=10)
    b1 = Simple_Bridge(parent=l1, child=l2)
    assert b1.parent == l1
    assert b1.child == l2


def test_assing_input():
    l1 = Simple_Layer(inputs_count=2, neuron_count=10)
    l2 = Simple_Layer(inputs_count=10, neuron_count=10)
    b1 = Simple_Bridge(parent=l1, child=l2)
    X = [1.5, 2.0]
    b1.X = X
    for x in b1.parent.X:
        assert x == [1.0] + X

def test_assing_ytrain():
    l1 = Simple_Layer(inputs_count=2, neuron_count=10)
    l2 = Simple_Layer(inputs_count=10, neuron_count=10)
    b1 = Simple_Bridge(parent=l1, child=l2)
    ytrain = list(range(10))
    b1.ytrain = ytrain
    for (neuron,desired_value) in zip(b1.child._items,ytrain):
        assert neuron.ytrain == desired_value

def test_forward_prop():
    l1 = Simple_Layer(inputs_count=2, neuron_count=10)
    l2 = Simple_Layer(inputs_count=10, neuron_count=10)
    b1 = Simple_Bridge(parent=l1, child=l2)
    b1.X = list(range(2))
    b1.forward()
    assert b1.child.out == b1.out

def test_train():
    l1 = Simple_Layer(inputs_count=2, neuron_count=2)
    l2 = Simple_Layer(inputs_count=3, neuron_count=2)
    b1 = Simple_Bridge(parent=l1, child=l2)
    def my_func(x,y):
        return (0.5*(y+x),0.2*(x+y))
    from random import random
    from functools import reduce
    cumulative_error = []
    from math import sin
    for i in range(500):

        x= sin(i /10.)
        y = sin(i /10.)
        # x = random()+0.001
        # y = random()+0.001
        b1.X = [x,y]
        o1,o2 = my_func(x,y)
        b1.child.ytrain = [o1,o2]
        cumulative_error.append(reduce((lambda x, y: x+(y)),b1.child.error, 0))
        b1.train()
    from matplotlib import pyplot as plt
    from scipy.ndimage.filters import gaussian_filter1d
    import os
    os.environ['DEBUG_WKNN'] = 'TRUE'
    ysmoothed = gaussian_filter1d(cumulative_error, sigma=6)
    if os.environ['DEBUG_WKNN'] == 'TRUE':
        plt.plot(cumulative_error, 'x')
        plt.plot(ysmoothed)
        plt.savefig('out.png')
    assert abs(ysmoothed[-1]) < 0.25

