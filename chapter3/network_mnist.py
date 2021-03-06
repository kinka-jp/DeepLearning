# -*- coding:utf-8 -*-
import sys,os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from chapter3Exercise import sigmoid,softmax
from PIL import Image
'''
normalize:Trueの場合、入力画像の値を０〜１となるように正規化する
          Flaseの場合、入力画像の値が0~255のままにする
flatten:Trueの場合、読み込んだ画像ファイルはnumpy配列として１列で格納される
one-hot-lable:Trueの場合、[0,0,1,0,0,0,0,0]のように回答を格納する。（１は正解）
            　Flaseの場合、単純に正解となるラベルの数値をかえる
'''
def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = softmax(a3)
    return y

x,t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y) #最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:"+str(float(accuracy_cnt)/len(x)))




