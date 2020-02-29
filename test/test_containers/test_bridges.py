from wknn.models.containers.simple_layer import Simple_Layer
from wknn.models.containers.simple_bridge import Simple_Bridge


def test_init():
    l1 = Simple_Layer(inputs_count=2, neuron_count=10)
    l2 = Simple_Layer(inputs_count=10, neuron_count=10)
    b1 = Simple_Bridge(parent=l1,child=l2)
    assert b1.parent == l1
    assert b1.child == l2