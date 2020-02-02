from wknn.models.containers.simple_layer import Simple_Layer


def test_sharing_common_X():
    l = Simple_Layer(inputs_count=2,neuron_count=10)
    l.X = [1.0,2.0]
    pass