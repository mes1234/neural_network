import math

def step_funct(x):
    if x > 0.:
        return 1.0
    else:
        return 0.0

def sigmoid_funct(x):
    return math.exp(x)/(1.+math.exp(x))    