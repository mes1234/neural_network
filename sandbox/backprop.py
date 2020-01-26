W = [0.2,0.3,0.1]
X = [1.0,-0.5,0.7]
yd = [0.,0.,0.]
bpe = [0.,0.,0.]
import math

def sum_y(W,X):
    res = 0
    for w,x in zip(W,X):
        res= res+w*x
    return res

def sigmoid_funct(x):
    return (mat h.exp(x)/(1.+math.exp(x))-0.5)*2

def derive(W,X,index,delta,funct,act_funct,yact):
    W[index]= W[index]+delta
    yp = act_funct(funct(W,X))
    return (yp-yact)/delta

def error_back(W,X,yd,error):
    res = [None] * len(W)
    for i in range(len(res)):
        res[i] = (error * yd[i])/sum(yd)
    return res

def adjust(W,yd,error):
    for i in range(len(W)):
        W[i]=W[i]+error*yd[i]
    return W

if __name__ == "__main__":
    error_list = []
    for j in range(100):
        y = sum_y(W,X)
        a = sigmoid_funct(y)
        error = -0.9 - a
        print(f"error = {error:0.4} a={a:0.4}") 
        for i in range(len(W)):
            yd[i]=derive(W.copy(),X,i,0.000001,sum_y,sigmoid_funct,a)
        bpe = error_back(W,X,yd,error)
        print(f"bpe bilanse is  {sum(bpe)-error}")
        W = adjust(W.copy(),yd,error)
        error_list.append(error)
    
    from matplotlib import pyplot as plt
    plt.plot(error_list)
    plt.grid()
    plt.show()