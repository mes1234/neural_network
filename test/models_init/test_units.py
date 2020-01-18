import pytest
from wknn.models.units.perceptron import Perceptron

def test_init():
    p =  Perceptron()
    assert isinstance(p,Perceptron)
