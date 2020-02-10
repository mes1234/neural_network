from wknn.models.containers.simple_layer import Simple_Layer


def test_sharing_common_X():
    l = Simple_Layer(inputs_count=2, neuron_count=10)
    X = [1.0, 0.1, 0.2]
    l.X = X
    for item in l._items:
        assert item.X == X


def test_setting_ytrain():
    l = Simple_Layer(inputs_count=2, neuron_count=10)
    ytrain = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    l.ytrain = ytrain
    for index, item in enumerate(l._items):
        assert item.ytrain == ytrain[index]


def test_calculating_error():
    l = Simple_Layer(inputs_count=3, neuron_count=2)
    l.X = [0.0, 0.0, 0.0]
    l.ytrain = [0.0, 0.0]
    error = l.error
    assert all(map(lambda x: x == -0.5, error)) == True
    pass


def test_train():
    def opt_funct(X):
        return [X[1], X[0]]

    from random import random
    from functools import reduce

    l = Simple_Layer(inputs_count=3, neuron_count=2)
    cumulative_error = []
    for i in range(5000):
        X = [1.0, random(), random()]
        l.X = X
        l.ytrain = opt_funct(X[1:])
        error = l.error
        cumulative_error.append(reduce((lambda x, y: x+abs(y)), error, 0))
        l.train()
    from matplotlib import pyplot as plt
    from scipy.ndimage.filters import gaussian_filter1d
    import os

    ysmoothed = gaussian_filter1d(cumulative_error, sigma=6)
    if os.environ['DEBUG_WKNN'] == 'TRUE':
        plt.plot(cumulative_error, 'x')
        plt.plot(ysmoothed)
        plt.show()
    assert ysmoothed[-1] < 0.1


def test_train_consts():
    def opt_funct(X):
        return [0.0001, 0.99]

    from random import random
    from functools import reduce

    l = Simple_Layer(inputs_count=3, neuron_count=2)
    cumulative_error = []
    for i in range(5000):
        X = [1.0, random(), random()]
        l.X = X
        l.ytrain = opt_funct(X[1:])
        error = l.error
        cumulative_error.append(reduce((lambda x, y: x+abs(y)), error, 0))
        l.train()
    from matplotlib import pyplot as plt
    from scipy.ndimage.filters import gaussian_filter1d
    import os

    ysmoothed = gaussian_filter1d(cumulative_error, sigma=6)
    if os.environ['DEBUG_WKNN'] == 'TRUE':
        plt.plot(cumulative_error, 'x')
        plt.plot(ysmoothed)
        plt.show()
    assert ysmoothed[-1] < 0.1


def test_response():
    from wknn.utilities.output_fuctions import sigmoid_funct
    from functools import reduce
    from random import random

    l = Simple_Layer(inputs_count=3, neuron_count=2)
    l.X = [1.0, random(), random()]
    out = l.out
    for index, output in enumerate(out):
        sum = reduce((lambda x, y: x+y[0]*y[1]),
                     zip(l._items[index].W, l._items[index].X), 0)
        assert output == sigmoid_funct(sum)


def test_back_propagating_error():
    from wknn.utilities.output_fuctions import sigmoid_funct
    from functools import reduce
    from random import random, randint

    neuron_count = randint(2, 1000)
    l = Simple_Layer(inputs_count=3, neuron_count=neuron_count)
    l.ytrain = [random() for x in range(neuron_count)]
    l.X = [1.0, random(), random()]
    bpe = l.back_prop_error
    bpe_raw = []
    for neuron in l._items:
        bpe_raw.append(list(neuron.back_prop_error))
    for i in range(3):
        assert bpe[i] == sum([row[i] for row in bpe_raw])
