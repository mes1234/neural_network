"""
Set of tests to test unit basic capability 
"""
import pytest
from wknn.models.units.perceptron import Perceptron
from wknn.models.units.dummy import Dummy
from wknn.interfaces.node import Node

def test_init():
    d =  Perceptron()
    assert isinstance(d,Dummy)
    assert isinstance(d,Node)

def test_XWytrain_setup():
    X =[1.,2.,3.]
    W =[4.,5.,6.]
    y_train = 0.0
    d = Perceptron()
    d.X = X
    d.W = W
    d.ytrain = y_train
    assert d.X == X
    assert d.W == W
    assert d.ytrain == y_train
    # with pytest.raises(ValueError):
    #     assert d.y == 1.0
    with pytest.raises(AttributeError):
        d.y = 1.0