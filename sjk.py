# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:38:06 2018

@author: 63159
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D
from keras.layers import LSTM, Flatten
from keras.regularizers import l2

# %%信道初始化
trainnumber = 50
testnumber = 50
region = 500
Ptdbm = 40
fmhz = 2400
Gt = 2.5
Gr = 2.5
Ndbm = -169
N = (10 ** (Ndbm / 10)) * 0.001
B = 5e6
M1 = 5
M2 = 15
M3 = 31
M = 31
internumber = 100
testinternumber = 10


# %%初始化发送机接收机
def linkpair(testnumber):
    t = np.random.random(testnumber) * 2 * np.pi - np.pi
    t = np.expand_dims(t, axis=1)
    x = np.cos(t)
    y = np.sin(t)
    i_set = np.arange(0, testnumber, 1)
    Tx = np.random.uniform(0, region, testnumber)
    Tx = np.expand_dims(Tx, axis=1)
    Ty = np.random.uniform(0, region, testnumber)
    Ty = np.expand_dims(Ty, axis=1)
    len = np.zeros((testnumber, 1))
    for i in i_set:
        len[i] = np.random.uniform(2, 65)
        x[i] = x[i] * len[i]
        y[i] = y[i] * len[i]
        Rx = Tx + x
        Ry = Ty + y
        for i in range(0, testnumber):
            if Rx[i] > region or Rx[i] < 0:
                Rx[i] = Tx[i] - 2 * x[i]
            if Ry[i] > region or Ry[i] < 0:
                Ry[i] = Ty[i] - 2 * y[i]
        T = np.vstack((Tx.T, Ty.T)).T
        R = np.vstack((Rx.T, Ry.T)).T
    return T, R, testnumber, len


# %%计算信道参数
def hparameter(T, R, testnumber):
    l = np.zeros((testnumber, testnumber))
    lenlog = np.zeros((testnumber, testnumber))
    H = np.zeros((testnumber, testnumber))
    for i in range(0, testnumber):
        for j in range(0, testnumber):
            l[i, j] = math.sqrt((T[i, 0] - R[j, 0]) ** 2 + (T[i, 1] - R[j, 1]) ** 2)
            lenlog[i, j] = 20 * math.log(l[i, j] / 1000, 10)
    L = 32.45 + 20 * math.log(fmhz, 10) - Gt - Gr
    L = np.ones((testnumber, testnumber)) * L + lenlog
    Prdbm = Ptdbm - L
    for i in range(0, testnumber):
        a = Prdbm[i] / 10
        H[i] = (10 ** a) * 0.001
    return H


# %%收发机分区，生成密度矩阵，每个link的送入CNN的密度图
def density(T, R, M, testnumber):
    grid1 = region / M
    tdensity = np.zeros((testnumber, 2), int)
    rdensity = np.zeros((testnumber, 2), int)
    Tdensity = np.zeros((testnumber, testnumber))
    Rdensity = np.zeros((testnumber, testnumber))
    for i in range(0, testnumber):
        tdensity[i, 0] = int(np.floor(T[i, 0] / grid1))
        tdensity[i, 1] = int(np.floor(T[i, 1] / grid1))
        Tdensity[tdensity[i, 0], tdensity[i, 1]] = Tdensity[tdensity[i, 0], tdensity[i, 1]] + 1
        rdensity[i, 0] = int(np.floor(R[i, 0] / grid1))
        rdensity[i, 1] = int(np.floor(R[i, 1] / grid1))
        Rdensity[rdensity[i, 0], rdensity[i, 1]] = Rdensity[rdensity[i, 0], rdensity[i, 1]] + 1
    return tdensity, rdensity, Tdensity, Rdensity


def zhouwei(x_density, y_density, Tdensity, Rdensity, M, M0, testnumber):
    Tzhouwei = np.zeros((testnumber, M, M), int)
    Rzhouwei = np.zeros((testnumber, M, M), int)
    a = (M - 1) / 2
    for i in range(0, testnumber):
        for j in range(0, M):
            for k in range(0, M):
                if (x_density[i, 0] - a + j) < 0 or (x_density[i, 1] - a + k) < 0 or (x_density[i, 0] - a + j) > M0 or (
                        x_density[i, 1] - a + k) > M0:
                    Tzhouwei[i, j, k] = 0
                else:
                    Tzhouwei[i, j, k] = Rdensity[int(x_density[i, 0] - a + j), int(x_density[i, 1] - a + k)]
                if (y_density[i, 0] - a + j) < 0 or (y_density[i, 1] - a + k) < 0 or (y_density[i, 0] - a + j) > M0 or (
                        y_density[i, 1] - a + k) > M0:
                    Rzhouwei[i, j, k] = 0
                else:
                    Rzhouwei[i, j, k] = Tdensity[int(y_density[i, 0] - a + j), int(y_density[i, 1] - a + k)]
    return Tzhouwei, Rzhouwei


# %%FPLinQ算法
def FPLinQ(number, H, N):
    maxiteration = 100
    wt = np.ones((number, 1))
    X = np.ones((number, 1))
    Z = np.zeros((number, 1))
    Y = np.zeros((number, 1))
    for k in range(0, maxiteration + 1):
        for i in range(0, number):
            hi = H[i, :]
            Z[i] = (H[i, i] * X[i]) / (np.dot(hi.T, X) - H[i, i] * X[i] + N)
            Y[i] = np.sqrt(wt[i] * (1 + Z[i]) * H[i, i] * X[i]) / float(np.dot(hi.T, X) + N)
            y = Y[i] * np.sqrt(wt[i] * (1 + Z[i]) * H[i, i]) / (np.dot(hi.T, np.multiply(Y, Y)))
            X[i] = min(1, y * y)
    for i in range(0, number):
        h = H[i, :]
        Q = 2 * Y[i] * np.sqrt(wt[i] * (1 + Z[i]) * H[i, i] * X[i]) - X[i] * (np.dot(hi.T, np.multiply(Y, Y)))
        if Q > 0:
            X[i] = 1
        else:
            X[i] = 0
    return X


# %%计算和速率
def delete(X, H):
    H1 = np.copy(H)
    for i in range(0, testnumber):
        if X[i] == 0:
            H1[i, :] = 0
            H1[:, i] = 0
    return H1


def rate(H, N, B):
    rate = np.zeros((testnumber, 1))
    snr = np.zeros((testnumber, 1))
    for i in range(0, testnumber):
        if H[i, i] != 0:
            hi = H[i, :]
            snr[i] = H[i, i] / (np.sum(hi) - H[i, i] + N)
            rate[i] = B * math.log(snr[i] + 1, 2)
    return rate


# %%存储数据集
def storagedataset(i, H, Tzhouwei1, Rzhouwei1, Tzhouwei2, Rzhouwei2, Tzhouwei3, Rzhouwei3, X, length, Taround1,
                   Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall):
    Taround1[i, :, :, :] = Tzhouwei1
    Taround2[i, :, :, :] = Tzhouwei2
    Taround3[i, :, :, :] = Tzhouwei3
    Raround1[i, :, :, :] = Rzhouwei1
    Raround2[i, :, :, :] = Rzhouwei2
    Raround3[i, :, :, :] = Rzhouwei3
    Xall[i, :, :] = X
    lengthall[i, :, :] = length
    Hall[i, :, :] = H
    return Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall


def getdata(number, testnumber):
    Taround1 = np.zeros((number, testnumber, M1, M1))
    Taround2 = np.zeros((number, testnumber, M2, M2))
    Taround3 = np.zeros((number, testnumber, M3, M3))
    Raround1 = np.zeros((number, testnumber, M1, M1))
    Raround2 = np.zeros((number, testnumber, M2, M2))
    Raround3 = np.zeros((number, testnumber, M3, M3))
    Hall = np.zeros((number, testnumber, testnumber))
    Sumrate = 0
    Sumrate1 = 0
    Xall = np.zeros((internumber, testnumber, 1))
    lengthall = np.zeros((internumber, testnumber, 1))
    for m in range(number):
        region = 500
        T, R, testnumber, length = linkpair(testnumber)
        H = hparameter(T, R, testnumber)
        X = FPLinQ(testnumber, H, N)
        x_density, y_density, Tdensity, Rdensity = density(T, R, M, testnumber)
        Tzhouwei1, Rzhouwei1 = zhouwei(x_density, y_density, Tdensity, Rdensity, M1, M, testnumber)
        Tzhouwei2, Rzhouwei2 = zhouwei(x_density, y_density, Tdensity, Rdensity, M2, M, testnumber)
        Tzhouwei3, Rzhouwei3 = zhouwei(x_density, y_density, Tdensity, Rdensity, M3, M, testnumber)
        Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall = storagedataset(m, H,
                                                                                                           Tzhouwei1,
                                                                                                           Rzhouwei1,
                                                                                                           Tzhouwei2,
                                                                                                           Rzhouwei2,
                                                                                                           Tzhouwei3,
                                                                                                           Rzhouwei3, X,
                                                                                                           length,
                                                                                                           Taround1,
                                                                                                           Taround2,
                                                                                                           Taround3,
                                                                                                           Raround1,
                                                                                                           Raround2,
                                                                                                           Raround3,
                                                                                                           Xall,
                                                                                                           lengthall,
                                                                                                           Hall)
        # %%评估FP,AA性能
        H1 = delete(X, H)
        Rate = np.zeros((testnumber, 1))
        Rate1 = np.zeros((testnumber, 1))
        Rate = rate(H1, N, B)
        Rate1 = rate(H, N, B)
        sumrate = np.sum(Rate) / 1e6
        sumrate1 = np.sum(Rate1) / 1e6
        #       plt.scatter(m,sumrate,s=15, color='g')
        #  plt.scatter(m,sumrate1,s=15, color='r')
        Sumrate = Sumrate + sumrate
        Sumrate1 = Sumrate1 + sumrate1
    return Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall, Sumrate, Sumrate1


# plt.figure(figsize=(13,10))
# %%CNN,DNN 神经网络搭建
reg = 0.01
Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall, Sumrate, Sumrate1 = getdata(
    internumber, trainnumber)
Taround1t, Taround2t, Taround3t, Raround1t, Raround2t, Raround3t, Xallt, lengthallt, Hallt, Sumratet, Sumrate1t = getdata(
    testinternumber, testnumber)
# CNN part
it1 = Input(shape=(M1, M1, 1), name='xd1')
it2 = Input(shape=(M2, M2, 1), name='xd2')
it3 = Input(shape=(M3, M3, 1), name='xd3')
it4 = Input(shape=(M1, M1, 1), name='yd1')
it5 = Input(shape=(M2, M2, 1), name='yd2')
it6 = Input(shape=(M3, M3, 1), name='yd3')
it7 = Input(shape=(1,), name='lenth')

x1 = Conv2D(1, kernel_size=M1, padding='same', activation='relu')(it1)
x2 = Conv2D(1, kernel_size=M2, padding='same', activation='relu')(it2)
x3 = Conv2D(1, kernel_size=M3, padding='same', activation='relu')(it3)
y1 = Conv2D(1, kernel_size=M1, padding='same', activation='relu')(it4)
y2 = Conv2D(1, kernel_size=M2, padding='same', activation='relu')(it5)
y3 = Conv2D(1, kernel_size=M3, padding='same', activation='relu')(it6)

x1 = Flatten()(x1)
x2 = Flatten()(x2)
x3 = Flatten()(x3)
y1 = Flatten()(y1)
y2 = Flatten()(y2)
y3 = Flatten()(y3)
x = keras.layers.concatenate([x1, x2, x3, y1, y2, y3, it7])
out_0 = Dense(21, activation='relu', W_regularizer=l2(reg))(x)
out_1 = Dense(21, activation='relu', W_regularizer=l2(reg))(out_0)
out_2 = Dense(1, activation='sigmoid', W_regularizer=l2(reg))(out_1)
model = Model(inputs=[it1, it2, it3, it4, it5, it6, it7], outputs=[out_2])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# %%训练网络
for i in range(0, internumber):
    if i != 0:
        model.load_weights('temple.h5')
    model.fit([np.expand_dims(Taround1[i, :, :, :], axis=-1), np.expand_dims(Taround2[i, :, :, :], axis=-1),
               np.expand_dims(Taround3[i, :, :, :], axis=-1), np.expand_dims(Raround1[i, :, :, :], axis=-1),
               np.expand_dims(Raround2[i, :, :, :], axis=-1), np.expand_dims(Raround3[i, :, :, :], axis=-1),
               lengthall[i, :, :]], Xall[i, :, :], epochs=1, batch_size=1)
    model.evaluate([np.expand_dims(Taround1[i, :, :, :], axis=-1), np.expand_dims(Taround2[i, :, :, :], axis=-1),
                    np.expand_dims(Taround3[i, :, :, :], axis=-1), np.expand_dims(Raround1[i, :, :, :], axis=-1),
                    np.expand_dims(Raround2[i, :, :, :], axis=-1), np.expand_dims(Raround3[i, :, :, :], axis=-1),
                    lengthall[i, :, :]], Xall[i, :, :], batch_size=1)
    model.save_weights('temple.h5')
# %%测试网络
out = np.zeros((testinternumber, testnumber, 1))
Sumrate1p = 0
for i in range(0, testinternumber):
    model.load_weights('temple.h5')
    res = model.predict([np.expand_dims(Taround1t[i, :, :, :], axis=-1), np.expand_dims(Taround2t[i, :, :, :], axis=-1),
                         np.expand_dims(Taround3t[i, :, :, :], axis=-1), np.expand_dims(Raround1t[i, :, :, :], axis=-1),
                         np.expand_dims(Raround2t[i, :, :, :], axis=-1), np.expand_dims(Raround3t[i, :, :, :], axis=-1),
                         lengthallt[i, :, :]], batch_size=1)
    out[i, :, :] = res
    predict1 = (res > 0.5)
    predict1 = (predict1 + 0)
    Hp = delete(predict1, Hallt[i, :, :])
    Ratep = np.zeros((testnumber, 1))
    Ratep = rate(Hp, N, B)
    Sumrate1p = Sumrate1p + Ratep
    Sumrate1p = np.sum(Sumrate1p) / 1e6
# %%性能输出
Sumratet = Sumratet / 100
Sumrate1t = Sumrate1t / 100
Sumrate1p = Sumrate1p / 100
print("average FP sumrate= ", Sumratet)
print("average AA sumrate1= ", Sumrate1t)
print("average CNN sumrate1= ", Sumrate1p)
"""
print("testnumber = ",testnumber)       
      %%%  plt.figure(figsize=(5,5),dpi=125)%%% 
plt.scatter(T[:,0], T[:,1], color='g')
plt.scatter(R[:,0], R[:,1], color='r')
plt.plot([T[:,0],R[:,0]],[T[:,1],R[:,1]], color='y')
plt.xlim(0,region)
plt.ylim(0,region)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random Scatter')
plt.grid(True)
plt.savefig('imag.png')
plt.show()
"""


































