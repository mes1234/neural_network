from wknn.models.units.perceptron import Perceptron
from itertools import tee

def test_run():
    p = Perceptron()
    p.X = [1.0,2.0,3.0]
    p.W = [0.1,0.2,0.3]
    assert p.y == 1.4

def test_train_step_function():
    from matplotlib import pyplot as plt
    p = Perceptron()
    p.W = [5.5,2.50,-2.01]
    X = {
        1: ([1.0,1.5,-0.5],0),
        2: ([1.0,0.5,-0.5],0),
        3: ([1.0,-0.5,-0.5],0),
        4: ([1.0,0.0,1.0],0),
        5: ([1.0,1.5,1.5],1),
        6: ([1.0,2.5,2.0],1),
        7: ([1.0,1.5,2.5],1),
        8: ([1.0,2.0,-1.0],1)
    }
    y = {}
    Xs = []
    Ys = []
    for ii in range(100):
        for k,v in X.items():
            p.X = v[0]
            p.ytrain =v[1]
            y[k] = {
                "output" : p.out,
                "error":p.error
            }
            Xs.append(range(-10,10,1))
            Ys.append(list(map(lambda x:(-p.W[0]-p.W[1]*x)/p.W[2],range(-10,10,1))))
            # for i in range(len(Xs)-1):
            #     plt.plot(Xs[i],Ys[i],"-",lw=0.5)
            plt.xlim(-4,4)
            plt.ylim(-4,4)
            plt.grid()
            p.train()
            pass
    m = map(lambda y:y[0][1:],filter(lambda x: x[1]==0,X.values()))
    p = map(lambda y:y[0][1:],filter(lambda x: x[1]==1,X.values()))
    m1,m2 = tee(m)
    p1,p2 = tee(p)
    import os
    if os.environ['DEBUG_WKNN'] == 'TRUE':
        plt.plot(Xs[-1],Ys[-1],"--",lw=2.0)
        plt.plot(Xs[0],Ys[0],"-",lw=2.0)
        plt.plot([x for x,y in m1],[y for x,y in m2],"x")
        plt.plot([x for x,y in p1],[y for x,y in p2],"s")
        plt.show()
    def check_error(x):
        if x == 0:
            return True
        else:
            return False
    assert all(map(lambda x: check_error(x['error']),y.values()))
    pass
