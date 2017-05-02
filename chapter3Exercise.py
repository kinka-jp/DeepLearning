# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)


'''出力関数'''
def identify_function(x):
    return x

'''
ソフトマックスの指数関数を行う際には、何らかの定数を足し算（もしくは引き算）しても結果は変わらない

関数の出力値の総和は１のため、各出力が「確率」として解釈することができる
'''
def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c)  #オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

'''
x = np.arange(-5.0,5.0,0.1)
y= sigmoid(x) #シグモイド関数
#y = step_function(x)　#ステップ関数
plt.plot(x,y)
plt.ylim(-0.1,1.1) #range of y axis
plt.show()
'''
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forward(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identify_function(a3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)    #(0.31682708,0.69627909)
