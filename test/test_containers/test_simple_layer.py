from wknn.models.containers.simple_layer import Simple_Layer


def test_sharing_common_X():
    l = Simple_Layer(inputs_count=2,neuron_count=10)
    X = [1.0,0.1,0.2]
    l.X = X
    for item in l._items:
        assert item.X == X 

def test_setting_ytrain():
    l = Simple_Layer(inputs_count=2,neuron_count=10)
    ytrain = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    l.ytrain = ytrain
    for index,item in enumerate(l._items):
        assert item.ytrain == ytrain[index]       