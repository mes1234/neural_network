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