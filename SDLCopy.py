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
import linklist


# %%信道初始化
trainnumber = 200   # 训练组链路数量
testnumber = 200   # 测试组链路数量
region = 1000
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
internumber = 100   # 地图数/组数
testinternumber = 100  # 验证集数量


# %%初始化发送机接收机
def linkpair(linknumber):
    # 返回T， R:存储坐标[(Tx, Ty)...],[(Rx, Ry)...], 链路数量， 各对链路的距离
    t = np.random.random(linknumber) * 2 * np.pi - np.pi  # （0 - pi）之间的值
    t = np.expand_dims(t, axis=1)   # 变成列矩阵
    x = np.cos(t)
    y = np.sin(t)
    i_set = np.arange(0, linknumber, 1)   # 1表示步长
    Tx = np.random.uniform(0, region, linknumber)
    Tx = np.expand_dims(Tx, axis=1)
    Ty = np.random.uniform(0, region, linknumber)
    Ty = np.expand_dims(Ty, axis=1)
    len = np.zeros((linknumber, 1))
    for i in i_set:
        len[i] = np.random.uniform(2, 65)
        x[i] = x[i] * len[i]
        y[i] = y[i] * len[i]
        Rx = Tx + x
        Ry = Ty + y
        for i in range(0, linknumber):
            if Rx[i] > region or Rx[i] < 0:
                Rx[i] = Tx[i] - 2 * x[i]
            if Ry[i] > region or Ry[i] < 0:
                Ry[i] = Ty[i] - 2 * y[i]
        T = np.vstack((Tx.T, Ty.T)).T  # vstack: 以行堆叠列表
        R = np.vstack((Rx.T, Ry.T)).T
    return T, R, linknumber, len


# %%计算信道参数
def hparameter(T, R, testnumber):
    # 返回N*N维信道状态矩阵H
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
    # 返回各链路i的收发器的格子坐标和权重 及 整张图的权重信息
    grid1 = region / M
    tdensity = np.zeros((testnumber, 3))
    rdensity = np.zeros((testnumber, 3))
    Tdensity = np.zeros((testnumber, testnumber))
    Rdensity = np.zeros((testnumber, testnumber))
    for i in range(0, testnumber):
        tdensity[i, 0] = int(np.floor(T[i, 0] / grid1))  # 取整，np.floor 向数轴左侧取点 （ T[i, 0]:第i个链路发射机的x坐标；
        tdensity[i, 1] = int(np.floor(T[i, 1] / grid1))  # #                              T[i, 1]:第i个链路发射机的y坐标 ）
        tdensity[i, 2] = 1          # 第三位存储链路权重
        # 地图共M*M个格子，每个格子内有n个收发机器
        Tdensity[int(tdensity[i, 0]), int(tdensity[i, 1])] = Tdensity[int(tdensity[i, 0]), int(tdensity[i, 1])] + 1   # 累加权重值
        rdensity[i, 0] = int(np.floor(R[i, 0] / grid1))
        rdensity[i, 1] = int(np.floor(R[i, 1] / grid1))
        rdensity[i, 2] = 1
        Rdensity[int(rdensity[i, 0]), int(rdensity[i, 1])] = Rdensity[int(rdensity[i, 0]), int(rdensity[i, 1])] + 1
    return tdensity, rdensity, Tdensity, Rdensity  # 大写N*N维，存整张图，小写存链路i的收发坐标及权重


def zhouwei(t_density, r_density, Tdensity, Rdensity, M0, M, linknumber):
    # 分别存储每个链路的发射机/接收机的周围环境（没有减掉自身）
    Tzhouwei = np.zeros((linknumber, M0, M0), int)
    Rzhouwei = np.zeros((linknumber, M0, M0), int)
    a = (M0 - 1) / 2
    for i in range(0, linknumber):
        for j in range(0, M0):
            for k in range(0, M0):
                if (t_density[i, 0] - a + j) < 0 or (t_density[i, 1] - a + k) < 0 or (t_density[i, 0] - a + j) > M or (
                        t_density[i, 1] - a + k) > M:
                    Tzhouwei[i, j, k] = 0
                else:
                    Tzhouwei[i, j, k] = Rdensity[int(t_density[i, 0] - a + j), int(t_density[i, 1] - a + k)]
                if int(t_density[i, 0] - a + k) == r_density[i, 0] and int(t_density[i, 1] - a + k) == r_density[i, 1]:
                    Tzhouwei[i, j, k] = Tzhouwei[i, j, k] - t_density[i, 2]
                if (r_density[i, 0] - a + j) < 0 or (r_density[i, 1] - a + k) < 0 or (r_density[i, 0] - a + j) > M or (
                        r_density[i, 1] - a + k) > M:
                    Rzhouwei[i, j, k] = 0
                else:
                    Rzhouwei[i, j, k] = Tdensity[int(r_density[i, 0] - a + j), int(r_density[i, 1] - a + k)]
                if int(r_density[i, 0] - a + j) == t_density[i, 0] and int(r_density[i, 1] - a + k) == t_density[i, 1]:
                    Rzhouwei[i, j, k] = Rzhouwei[i, j, k] - r_density[i, 2]
    return Tzhouwei, Rzhouwei   # 三维数组，第一维表示链路，二三维存储环境


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
def storagedataset(i, H, tdensity, rdensity, tden, rden, Tzhouwei1, Rzhouwei1, Tzhouwei2, Rzhouwei2, Tzhouwei3, Rzhouwei3, X, length, Taround1,
                   Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall):
    tden[i, :, :] = tdensity
    rden[i, :, :] = rdensity
    Taround1[i, :, :, :] = Tzhouwei1
    Taround2[i, :, :, :] = Tzhouwei2
    Taround3[i, :, :, :] = Tzhouwei3
    Raround1[i, :, :, :] = Rzhouwei1
    Raround2[i, :, :, :] = Rzhouwei2
    Raround3[i, :, :, :] = Rzhouwei3
    Xall[i, :, :] = X
    lengthall[i, :, :] = length
    Hall[i, :, :] = H                    # 每个数据再加一维，在第一维形成堆叠。第一维表示轮数
    return tden, rden, Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall


def getdata(mapnumber, testnumber):
    Taround1 = np.zeros((mapnumber, testnumber, M1, M1))
    Taround2 = np.zeros((mapnumber, testnumber, M2, M2))
    Taround3 = np.zeros((mapnumber, testnumber, M3, M3))
    Raround1 = np.zeros((mapnumber, testnumber, M1, M1))
    Raround2 = np.zeros((mapnumber, testnumber, M2, M2))
    Raround3 = np.zeros((mapnumber, testnumber, M3, M3))
    tden = np.zeros((mapnumber, testnumber, 3))
    rden = np.zeros((mapnumber, testnumber, 3))
    Hall = np.zeros((mapnumber, testnumber, testnumber))
    Sumrate = 0
    SumrateAA = 0
    Xall = np.zeros((internumber, testnumber, 1))
    lengthall = np.zeros((internumber, testnumber, 1))
    for m in range(mapnumber):
        T, R, testnumber, length = linkpair(testnumber)
        H = hparameter(T, R, testnumber)
        X = FPLinQ(testnumber, H, N)
        tdensity, rdensity, Tdensity, Rdensity = density(T, R, M, testnumber)
        Tzhouwei1, Rzhouwei1 = zhouwei(tdensity, rdensity, Tdensity, Rdensity, M1, M, testnumber)
        Tzhouwei2, Rzhouwei2 = zhouwei(tdensity, rdensity, Tdensity, Rdensity, M2, M, testnumber)
        Tzhouwei3, Rzhouwei3 = zhouwei(tdensity, rdensity, Tdensity, Rdensity, M3, M, testnumber)
        tden, rden, Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall= storagedataset(
                                                                                                           m, H,
                                                                                                           tdensity,
                                                                                                           rdensity,
                                                                                                           tden,
                                                                                                           rden,
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
        RateAA = np.zeros((testnumber, 1))
        Rate = rate(H1, N, B)
        RateAA = rate(H, N, B)
        sumrate = np.sum(Rate) / 1e6
        sumrateAA = np.sum(RateAA) / 1e6
        #       plt.scatter(m,sumrate,s=15, color='g')
        #  plt.scatter(m,sumrate1,s=15, color='r')
        Sumrate = Sumrate + sumrate
        SumrateAA = SumrateAA + sumrateAA
    return tden, rden, Taround1, Taround2, Taround3,\
           Raround1, Raround2, Raround3, Xall, lengthall, Hall, Sumrate, SumrateAA
    # #  存储了m轮数据，一轮数据代表一张拓扑图
# Taround1第一维表示数据组序号，第二维表示链路序号，三四维表示第一个size的kernel的周围环境


def updatetrmap(predict, tden, rden, lens, testnumber, M1, M2, M3):
    # 直接返回七个输入，但是输入tden, rden, len暂时无法获取
    tdenx = np.copy(tden)
    rdenx = np.copy(rden)
    len = np.copy(lens)
    Tdx = np.zeros((testnumber, testnumber))
    Rdx = np.zeros((testnumber, testnumber))
    for j in range(0, testnumber):
        tdenx[j, 2] = predict[j]
        rdenx[j, 2] = predict[j]
        Tdx[int(tdenx[j, 0]), int(tdenx[j, 1])] = Tdx[int(tdenx[j, 0]), int(tdenx[j, 1])] + predict[j]
        Rdx[int(rdenx[j, 0]), int(rdenx[j, 1])] = Tdx[int(rdenx[j, 0]), int(rdenx[j, 1])] + predict[j]
    Tzw1, Rzw1 = zhouwei(tdenx, rdenx, Tdx, Rdx, M1, M, testnumber)
    Tzw2, Rzw2 = zhouwei(tdenx, rdenx, Tdx, Rdx, M2, M, testnumber)
    Tzw3, Rzw3 = zhouwei(tdenx, rdenx, Tdx, Rdx, M3, M, testnumber)
    return Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, len


def trainmodel(model, Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length, X):
    model.fit([np.expand_dims(Tzw1, axis=-1), np.expand_dims(Tzw2, axis=-1),
               np.expand_dims(Tzw3, axis=-1), np.expand_dims(Rzw1, axis=-1),
               np.expand_dims(Rzw2, axis=-1), np.expand_dims(Rzw3, axis=-1),
               length], X, epochs=1, batch_size=1)
    model.evaluate([np.expand_dims(Tzw1, axis=-1), np.expand_dims(Tzw2, axis=-1),
                    np.expand_dims(Tzw3, axis=-1), np.expand_dims(Rzw1, axis=-1),
                    np.expand_dims(Rzw2, axis=-1), np.expand_dims(Rzw3, axis=-1),
                    length], X, batch_size=1)
    model.save_weights('temple.h5')
    model.load_weights('temple.h5')
    #  predict函数按batch获得输入数据对应的输出
    xt = model.predict(
        [np.expand_dims(Tzw1, axis=-1), np.expand_dims(Tzw2, axis=-1),
         np.expand_dims(Tzw3, axis=-1), np.expand_dims(Rzw1, axis=-1),
         np.expand_dims(Rzw2, axis=-1), np.expand_dims(Rzw3, axis=-1),
         length], batch_size=1)  # axis=-1表示最后一维被扩展一维
    #    out[i, :, :] = res
    # return continuous value
    # xt = (xt > 0.5)  # return True or False
    # xt = (xt + 0)  # turn Bool into Int
    return xt, model


# plt.figure(figsize=(13,10))
# %%CNN,DNN 神经网络搭建
reg = 0.01
tden, rden, Taround1, Taround2, Taround3, Raround1, Raround2, Raround3,\
    Xall, lengthall, Hall, Sumrate, SumrateAA = getdata(internumber, trainnumber)   # internumber:地图数 ， trainnumber: 链路数
tdent, rdent, Taround1t, Taround2t, Taround3t, Raround1t, Raround2t, Raround3t, Xallt, lengthallt, Hallt, Sumratet,\
    SumrateAAt = getdata(testinternumber, testnumber)  # testinternumber:测试组地图数， testnumber: 测试集链路数
# CNN part
it1 = Input(shape=(M1, M1, 1), name='xd1')
it2 = Input(shape=(M2, M2, 1), name='xd2')
it3 = Input(shape=(M3, M3, 1), name='xd3')
it4 = Input(shape=(M1, M1, 1), name='yd1')
it5 = Input(shape=(M2, M2, 1), name='yd2')
it6 = Input(shape=(M3, M3, 1), name='yd3')  # 输入
it7 = Input(shape=(1,), name='lenth')       # 输入1维，不指定个数

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
out_0 = Dense(21, activation='relu', W_regularizer=l2(reg))(x)   # regualrizer正则项
out_1 = Dense(21, activation='relu', W_regularizer=l2(reg))(out_0)
out_2 = Dense(1, activation='sigmoid', W_regularizer=l2(reg))(out_1)
model = Model(inputs=[it1, it2, it3, it4, it5, it6, it7], outputs=[out_2])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# %%训练网络
for i in range(0, internumber):   # internumber 地图数/组数
    if i != 0:   # 用一个链路的数据训练一个模型，并存储，适用于所有链路
        model.load_weights('temple.h5')  # Taround1第一维表示数据组序号，第二维表示链路序号，三四维表示第一个size的kernel的周围环境
    predict11, model = trainmodel(model, Taround1[i, :, :, :], Raround1[i, :, :, :], Taround2[i, :, :, :], Raround2[i, :, :, :], Taround3[i, :, :, :], Raround3[i, :, :, :],
                                  lengthall[i, :, :], Xall[i, :, :])
    model.load_weights('temple.h5')
    Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length = updatetrmap(predict11, tden[i, :, :], rden[i, :, :],
                                                             lengthall[i, :, :],
                                                             testnumber, M1, M2, M3)
    predict2, model = trainmodel(model, Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length, Xall[i, :, :])
    model.load_weights('temple.h5')
    Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length = updatetrmap(predict2, tden[i, :, :], rden[i, :, :], lengthall[i, :, :],
                                                             testnumber, M1, M2, M3)
    predict3, model = trainmodel(model, Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length, Xall[i, :, :])
    model.load_weights('temple.h5')
    Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length = updatetrmap(predict11, tden[i, :, :], rden[i, :, :],
                                                             lengthall[i, :, :], testnumber, M1, M2, M3)
    predict2, model = trainmodel(model, Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length, Xall[i, :, :])
    model.load_weights('temple.h5')
    Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length = updatetrmap(predict11, tden[i, :, :], rden[i, :, :],
                                                             lengthall[i, :, :], testnumber, M1, M2, M3)
    predict2, model = trainmodel(model, Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length, Xall[i, :, :])
    model.load_weights('temple.h5')
    Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length = updatetrmap(predict11, tden[i, :, :], rden[i, :, :],
                                                             lengthall[i, :, :], testnumber, M1, M2, M3)
    predict2, model = trainmodel(model, Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length, Xall[i, :, :])

# %%测试网络
# out = np.zeros((testinternumber, testnumber, 1))
Sumrate1p = 0
for i in range(0, testinternumber):   # testinternumber = 10, testnumber = 50
    model.load_weights('temple.h5')
#  predict函数按batch获得输入数据对应的输出
    res = model.predict([np.expand_dims(Taround1t[i, :, :, :], axis=-1), np.expand_dims(Taround2t[i, :, :, :], axis=-1),
                        np.expand_dims(Taround3t[i, :, :, :], axis=-1), np.expand_dims(Raround1t[i, :, :, :], axis=-1),
                        np.expand_dims(Raround2t[i, :, :, :], axis=-1), np.expand_dims(Raround3t[i, :, :, :], axis=-1),
                        lengthallt[i, :, :]], batch_size=1)  # axis=-1表示最后一维被扩展一维
#    out[i, :, :] = res
    predict1 = (res > 0.5)   # return True or False
    predict1 = (predict1 + 0)  # turn Bool into Int
    Hp = delete(predict1, Hallt[i, :, :])
    if i == 3:
        print('test-NO.3:', predict1)
    Ratep = np.zeros((testnumber, 1))
    Ratep = rate(Hp, N, B)
    Sumrate1p = Sumrate1p + Ratep
Sumrate1p = np.sum(Sumrate1p) / 1e6
# %%性能输出
Sumratet = Sumratet / testinternumber
Sumrate1t = SumrateAAt / testinternumber
Sumrate1p = Sumrate1p / testinternumber
print('N = ', trainnumber)
print("Average FP sumrate= ", Sumratet, 'MBps')
print("Average All-Active sumrate1= ", Sumrate1t, 'MBps')
print("Average CNN sumrate1= ", Sumrate1p, 'MBps')
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
不传输2进制 X， 反馈连续的 X
最终目标：更新密度图！
密度更新：tdensity[i, 0/1]仅仅存储了位置信息，需要扩大一维，存储权值信息，在更新zhouwei函数时，+1操作改成+权值操作

"""







