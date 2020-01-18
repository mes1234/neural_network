from wknn.models.units.perceptron import Perceptron


def test_run():
    p = Perceptron()
    p.X = [1.0,2.0,3.0]
    p.W = [0.1,0.2,0.3]
    assert p.y == 1.4
def test_train():
    p = Perceptron()
    p.X = [1.0,2.0,3.0]
    p.W = [0.1,0.2,0.3]
    a= p.y
    p.ytrain = 1.0
    p.train()
    a = p.y
    b = p.y
    w = list(p.W)
    pass

