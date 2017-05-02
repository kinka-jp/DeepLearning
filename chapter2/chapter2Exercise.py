# -*- coding:utf-8 -*-
import numpy as np

'''perceptron/パーセプトロン/感知器（神经信号传递原理）'''

def AND_old(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

'''重さ：w、バイアス：b'''
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(x*w)+b
    if tmp <=0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])  #重さとバイアスだけがアandと違う
    b = 0.7
    tmp = np.sum(x*w)+b
    if tmp <=0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.1
    tmp = np.sum(x*w)+b
    if tmp <=0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

print("AND calculation:")
print AND(0,0)
print AND(0,1)
print AND(1,0)
print AND(1,1)

print("NAND calculation:")
print NAND(0,0)
print NAND(0,1)
print NAND(1,0)
print NAND(1,1)

print("OR calculation:")
print OR(0,0)
print OR(0,1)
print OR(1,0)
print OR(1,1)

print("XOR calculation:")
print XOR(0,0)
print XOR(0,1)
print XOR(1,0)
print XOR(1,1)