# -*- coding: utf-8 -*-
"""
Convolution is an operation on two functions f and g, which produces a third function that can be interpreted as a
modified ("filtered") version of f. In this interpretation we call g the filter. If f is defined on a spatial variable
like x rather than a time variable like t, we call the operation spatial convolution.
"""
import numpy as np
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt
import math
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D
from keras.layers import LSTM, Flatten
from keras.regularizers import l2
import linklist


# $$$$$$$$$ 初始化  $$$$$$$$$$$
trainnumber = 50   # 训练组链路数量
region = 500
Ptdbm = 40
fmhz = 2400
Gt = 2.5
Gr = 2.5
Ndbm = -169
Noise = (10 ** (Ndbm / 10)) * 0.001
B = 5e6
M1 = 5
M2 = 15
M3 = 31
M = 31
internumber = 500   # 地图数/组数
testMAPnumber = 80   # 测试集数量


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
    return T, R, len


# %%计算信道参数
def hparameter(T, R, linknumber):
    # 返回N*N维信道状态矩阵H
    h = np.zeros([linknumber, linknumber], dtype=float)
    H = np.zeros([linknumber, linknumber], dtype=float)
    for i in range(0, linknumber):
        for j in range(0, linknumber):
            h[i, j] = 32.45 + 20 * math.log10(np.sqrt((T[j, 0] - R[i, 0]) ** 2 + (T[j, 1] - R[i, 1]) ** 2) / 1000) \
                      + 20 * math.log(fmhz, 10) - Gt - Gr
    # 当Pi=1时， H[ii]数值上等于pi*h[i,j]^2,
    Prdbm = Ptdbm - h  # (L=Pt/Pr)
    for i in range(0, linknumber):
        a = Prdbm[i] / 10
        H[i] = (10 ** a) * 0.001
    return H


# %%收发机分区，生成密度矩阵，每个link的送入CNN的密度图
# 返回各链路i的收发器格子坐标和权重 以及 整张图的密度信息
# 大写N*N维，存整张图密度，小写存链路i格子坐标及权重
def density(T, R, M, testnumber):
    # 返回各链路i的收发器格子坐标和权重 以及 整张图的密度信息
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
    return tdensity, rdensity, Tdensity, Rdensity  # 大写N*N维，存整张图密度信息，小写存链路i格子坐标及权重


# 返回每个发射接收机的周围环境信息（准备做卷积）
def zhouwei(t_density, r_density, Tdensity, Rdensity, M0, M, linknumber):
    # 返回每个发射接收机的周围环境信息（准备做卷积）
    # 分别存储每个链路的发射机/接收机的周围环境（减掉自身）
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


# %%FPLinQ算法， 返回X，表示每个链路的激活状态信息
def FPLinQ(x, y, z, H, linknumber, Noise):
    maxiteration = 100
    w = 1
    p = 1
    for t in range(0, maxiteration):
        for i in range(0, linknumber):
            z[i] = (H[i, i] * x[i]) / (np.dot(H[i, :].T, x) - H[i, i] * x[i] + Noise)  # numpyarray.T: 转置
            y[i] = math.sqrt(w * (1 + z[i]) * H[i, i] * p * x[i]) / (np.dot(H[i, :].T, x) + Noise)
            xxx = (y[i] * np.sqrt(w * (1 + z[i]) * H[i, i] * p)) / (np.dot(H[:, i].T, (np.multiply(y, y))))
            x[i] = min(1, xxx * xxx)
    for i in range(0, linknumber):
        Q = 2 * y[i] * math.sqrt(w * (1 + z[i]) * H[i, i]) - (np.dot(H[:, i].T, np.multiply(y, y)))
        if Q > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x


# %%清除（置0）被关闭（x为0）的链路的信道信息，以及跟这些链路相关的交叉（干扰）信道信息，返回余下信道状态信息
def updateH(x, H, linknumber):
    H1 = np.copy(H)
    for i in range(0, linknumber):
        if x[i] == 0:
            H1[i, :] = 0
            H1[:, i] = 0
    return H1


#  计算和速率
#  返回rate,存储每个链路的速率
def rate(x, H, Noise, linknumber):
    rate = np.zeros([linknumber, 1], dtype=float)
    actlinknum = 0
    snr = np.zeros((linknumber, 1))
    sum1 = 0
    for i in range(0, linknumber):
        if H[i, i] != 0:
            actlinknum = actlinknum + 1
            for j in range(0, linknumber):
                sum1 = sum1 + H[i, j] * x[j]
            snr[i] = H[i, i] / (sum1 - H[i, i] * x[i] + Noise)
            sum1 = 0
            rate[i] = 5 * np.log2(1 + snr[i])
    sumrate = sum(rate)
    return sumrate


# %%存储数据集
# 将m组图的全部信息存储在一起
def storagedataset(i, H, tdensity, rdensity, tden, rden, Tzhouwei1, Rzhouwei1, Tzhouwei2, Rzhouwei2, Tzhouwei3, Rzhouwei3, X, length, Taround1,
                   Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall):
    tden[i, :, :] = tdensity    # 第i组数据的【所有链路】的格子坐标
    rden[i, :, :] = rdensity    #
    Taround1[i, :, :, :] = Tzhouwei1  # 第i组数据的【...】的周围信息
    Taround2[i, :, :, :] = Tzhouwei2
    Taround3[i, :, :, :] = Tzhouwei3
    Raround1[i, :, :, :] = Rzhouwei1
    Raround2[i, :, :, :] = Rzhouwei2
    Raround3[i, :, :, :] = Rzhouwei3
    Xall[i, :, :] = X                 # 第i组数据的【...】的链路状态信息
    lengthall[i, :, :] = length        # 第i组数据的【...】链路间距信息, 不存储交叉链路
    Hall[i, :, :] = H                    # 第i组数据的【...】信道状态信息，存储了交叉信道
    return tden, rden, Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall


def getdata(mapnumber, linknumber):
    Taround1 = np.zeros((mapnumber, linknumber, M1, M1))
    Taround2 = np.zeros((mapnumber, linknumber, M2, M2))
    Taround3 = np.zeros((mapnumber, linknumber, M3, M3))
    Raround1 = np.zeros((mapnumber, linknumber, M1, M1))
    Raround2 = np.zeros((mapnumber, linknumber, M2, M2))
    Raround3 = np.zeros((mapnumber, linknumber, M3, M3))
    tden = np.zeros((mapnumber, linknumber, 3))
    rden = np.zeros((mapnumber, linknumber, 3))
    Hall = np.zeros((mapnumber, linknumber, linknumber))
    Sumrate = 0
    SumrateAA = 0
    Xall = np.zeros((internumber, linknumber, 1))
    lengthall = np.zeros((internumber, linknumber, 1))
    for m in range(0, mapnumber):
        #  ===随机初始化 =====
        xx = np.random.randint(0, 2, (linknumber, 1))  # [low, high).
        x = np.ones([linknumber, 1], dtype=float)
        x1 = np.ones([linknumber, 1], dtype=float)  # AA
        z = np.zeros([linknumber, 1], dtype=float)
        y = np.zeros([linknumber, 1], dtype=float)
        #        xr = np.random.randint(0, 2, [linknumber, 1]) # random
        #        for i in range(0, linknumber):
        #            if xx[i] == 0:
        #                x[i] = 0
        T, R, length = linkpair(linknumber)
        H = hparameter(T, R, linknumber)
        x = FPLinQ(x, y, z, H, linknumber, Noise)
        # $$$$$$$$ 评估FP,AA性能
        H1 = updateH(x, H, linknumber)    # FP算法-更新信道状态信息
        RateFP = rate(x, H1, Noise, linknumber)
        RateAA = rate(x1, H, Noise, linknumber)
        Sumrate = Sumrate + RateFP      # FP算法和速率
        SumrateAA = SumrateAA + RateAA  # AA算法和速率
        # =================================存储数据================================================ #
        tdensity, rdensity, Tdensity, Rdensity = density(T, R, M, linknumber)
        Tzhouwei1, Rzhouwei1 = zhouwei(tdensity, rdensity, Tdensity, Rdensity, M1, M, linknumber)
        Tzhouwei2, Rzhouwei2 = zhouwei(tdensity, rdensity, Tdensity, Rdensity, M2, M, linknumber)
        Tzhouwei3, Rzhouwei3 = zhouwei(tdensity, rdensity, Tdensity, Rdensity, M3, M, linknumber)
        tden, rden, Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall = storagedataset(
            m, H, tdensity, rdensity, tden, rden, Tzhouwei1, Rzhouwei1, Tzhouwei2, Rzhouwei2, Tzhouwei3, Rzhouwei3, x,
            length, Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall)
        print(m, 'and fp rates now is', Sumrate, '\t', 'and AA rates is', SumrateAA)
        # ======================================================================================= #
    return tden, rden, Taround1, Taround2, Taround3,\
           Raround1, Raround2, Raround3, Xall, lengthall, Hall, Sumrate, SumrateAA
    # #  存储了m轮数据，一轮数据代表一张拓扑图
# Taround1第一维表示数据组序号，第二维表示链路序号，三四维表示第一个size的kernel的周围环境


def creat_model(reg):
    in1 = Input(shape=(M1, M1, 1))
    in2 = Input(shape=(M2, M2, 1))
    in3 = Input(shape=(M3, M3, 1))
    in4 = Input(shape=(M1, M1, 1))
    in5 = Input(shape=(M2, M2, 1))
    in6 = Input(shape=(M3, M3, 1))  # 输入
    in7 = Input(shape=(1,))  # 输入1维，不指定个数

    in_1 = Conv2D(1, kernel_size=M1, padding='same', activation='relu')(in1)   # 1代表通道1
    in_2 = Conv2D(1, kernel_size=M2, padding='same', activation='relu')(in2)
    in_3 = Conv2D(1, kernel_size=M3, padding='same', activation='relu')(in3)
    in_4 = Conv2D(1, kernel_size=M1, padding='same', activation='relu')(in4)
    in_5 = Conv2D(1, kernel_size=M2, padding='same', activation='relu')(in5)
    in_6 = Conv2D(1, kernel_size=M3, padding='same', activation='relu')(in6)

    x1 = Flatten()(in_1)
    x2 = Flatten()(in_2)
    x3 = Flatten()(in_3)
    y1 = Flatten()(in_4)
    y2 = Flatten()(in_5)
    y3 = Flatten()(in_6)
    x = keras.layers.concatenate([x1, x2, x3, y1, y2, y3, in7])
    out_0 = Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  # regualrizer正则项
    out_1 = Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.01))(out_0)
    out_2 = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(out_1)
    model = Model(inputs=[in1, in2, in3, in4, in5, in6, in7], outputs=[out_2])
    model.compile(optimizer=optimizers.RMSprop(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])  # lr = 0.001参数怎么调
    model.summary()
    return model


def updatemap(predict, tden, rden, lens, linknumber, M1, M2, M3):
    # 直接返回七个输入，但是输入tden, rden, len暂时无法获取
    tdenx = np.copy(tden)
    rdenx = np.copy(rden)
    len = np.copy(lens)
    Tdx = np.zeros((linknumber, linknumber))
    Rdx = np.zeros((linknumber, linknumber))
    for j in range(0, linknumber):
        tdenx[j, 2] = predict[j]
        rdenx[j, 2] = predict[j]
        Tdx[int(tdenx[j, 0]), int(tdenx[j, 1])] = Tdx[int(tdenx[j, 0]), int(tdenx[j, 1])] + tdenx[j, 2]
        Rdx[int(rdenx[j, 0]), int(rdenx[j, 1])] = Tdx[int(rdenx[j, 0]), int(rdenx[j, 1])] + rdenx[j, 2]
    Tzw1, Rzw1 = zhouwei(tdenx, rdenx, Tdx, Rdx, M1, M, linknumber)
    Tzw2, Rzw2 = zhouwei(tdenx, rdenx, Tdx, Rdx, M2, M, linknumber)
    Tzw3, Rzw3 = zhouwei(tdenx, rdenx, Tdx, Rdx, M3, M, linknumber)
    return Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, len


def trainmodel(model, Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length, X):
    tdenv, rdenv, Taround1v, Taround2v, Taround3v, Raround1v, Raround2v, Raround3v, \
    Xallv, lengthallv, Hallv, Sumratev, SumrateAAv = getdata(1, 50)  # 生成交叉验证集
    model.fit([np.expand_dims(Tzw1, axis=-1), np.expand_dims(Tzw2, axis=-1),
               np.expand_dims(Tzw3, axis=-1), np.expand_dims(Rzw1, axis=-1),
               np.expand_dims(Rzw2, axis=-1), np.expand_dims(Rzw3, axis=-1),
               length], X, epochs=4, batch_size=10, validation_data=(
                   [np.expand_dims(Taround1v[0, :, :, :], axis=-1), np.expand_dims(Taround2v[0, :, :, :], axis=-1),
                    np.expand_dims(Taround3v[0, :, :, :], axis=-1), np.expand_dims(Raround1v[0, :, :, :], axis=-1),
                    np.expand_dims(Raround2v[0, :, :, :], axis=-1), np.expand_dims(Raround3v[0, :, :, :], axis=-1),
                    lengthallv[0, :, 0]], Xallv[1, :, 0]))
    # epochs = 8, batch-size = 10训练出过good model, feedback 5次
    # epochs = 4, batch-size = 10训练出过good model, feedback 3次
#    model.evaluate([np.expand_dims(Tzw1, axis=-1), np.expand_dims(Tzw2, axis=-1),
#                    np.expand_dims(Tzw3, axis=-1), np.expand_dims(Rzw1, axis=-1),
#                    np.expand_dims(Rzw2, axis=-1), np.expand_dims(Rzw3, axis=-1),
#                    length], X, batch_size=1)
    #  predict函数按batch获得输入数据对应的输出
    xt = model.predict(
        [np.expand_dims(Tzw1, axis=-1), np.expand_dims(Tzw2, axis=-1),
         np.expand_dims(Tzw3, axis=-1), np.expand_dims(Rzw1, axis=-1),
         np.expand_dims(Rzw2, axis=-1), np.expand_dims(Rzw3, axis=-1),
         length], batch_size=1)  # axis=-1表示最后一维被扩展一维
    # return continuous value
    return xt, model


reg = 0.01
model = creat_model(reg)

# $$$$$ 训练网络  $$$$$$$
xold = np.zeros((trainnumber, 1))      # 训练前的x
xupdate = np.zeros((trainnumber, 1))   # 送入更新地图的x
xpred = np.zeros((trainnumber, 1))     # 训练后的x
tden, rden, Taround1, Taround2, Taround3, Raround1, Raround2, Raround3,\
    Xall, lengthall, Hall, Sumrate, SumrateAA = getdata(internumber, trainnumber)   # internumber:地图数 ， trainnumber: 链路数
for i in range(0, internumber):   # internumber 地图数/组数
    print('Now is map:', i)
    xpred, model = trainmodel(model, Taround1[i, :, :, :], Raround1[i, :, :, :], Taround2[i, :, :, :], Raround2[i, :, :, :], Taround3[i, :, :, :], Raround3[i, :, :, :],
                                  lengthall[i, :, 0], Xall[i, :, 0])
    xold = Xall[i, :, :]
    for five in range(0, 3):
        # 每个链路的output有50%的机率feedback
        for fb in range(0, trainnumber):
            a = np.random.uniform(0, 1)
            if a > 0.5:
                xupdate[fb, 0] = xpred[fb, 0]
            if a <= 0.5:
                xupdate[fb, 0] = xold[fb, 0]
        # ================================== #
        Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length = updatemap(xupdate, tden[i, :, :], rden[i, :, :],
                                                               lengthall[i, :, 0], trainnumber, M1, M2, M3)
        xpred, model = trainmodel(model, Tzw1, Rzw1, Tzw2, Rzw2, Tzw3, Rzw3, length, Xall[i, :, 0])
        xold = xupdate
model.save('20190322-2.h5')
# $$$$$ 测试网络  $$$$$$$

step = 5
axisX = np.zeros(step)
sumfp = np.zeros(step)
sumdl = np.zeros(step)
sumaa = np.zeros(step)
for nk in range(1, step):
    testnumber = 50*nk
    tdent, rdent, Taround1t, Taround2t, Taround3t, Raround1t, Raround2t, Raround3t, Xallt, lengthallt, Hallt, Sumratet,\
        SumrateAAt = getdata(testMAPnumber, testnumber)  # testinternumber:测试组地图数， testnumber: 测试集链路数
    Sumrate1p = 0
    for i in range(0, testMAPnumber):   # testinternumber = 10, testnumber = 50
        model.load_weights('20190322-2.h5')
        # predict函数按batch获得输入数据对应的输出
        res = model.predict([np.expand_dims(Taround1t[i, :, :, :], axis=-1), np.expand_dims(Taround2t[i, :, :, :], axis=-1),
                            np.expand_dims(Taround3t[i, :, :, :], axis=-1), np.expand_dims(Raround1t[i, :, :, :], axis=-1),
                            np.expand_dims(Raround2t[i, :, :, :], axis=-1), np.expand_dims(Raround3t[i, :, :, :], axis=-1),
                            lengthallt[i, :, :]], batch_size=1)  # axis=-1表示最后一维被扩展一维
        #    out[i, :, :] = res
        predict1 = (res > 0.5)  # return True or False
        predict1 = (predict1 + 0)  # turn Bool into Int
        Hp = updateH(predict1, Hallt[i, :, :], testnumber)
        if i == 4:
            print('test-Map_NO.3:', predict1.T)
        RateDL = rate(predict1, Hp, Noise, testnumber)
        Sumrate1p = Sumrate1p + RateDL
    # $$$$$$$output
    Sumratet = Sumratet / testMAPnumber
    Sumrate1t = SumrateAAt / testMAPnumber
    Sumrate1p = Sumrate1p / testMAPnumber
    axisX[nk] = testnumber
    sumdl[nk] = Sumrate1p
    sumfp[nk] = Sumratet
    sumaa[nk] = Sumrate1t
    print('test-number = ', testnumber)
    print('FP-sum-rate= ', Sumratet, 'MBps')
    print('All-Active sum-rate1= ', Sumrate1t, 'MBps')
    print('DL-sum-rate1= ', Sumrate1p, 'MBps')
plt.figure(1)
plt.plot(axisX, sumdl, '-*', label='DL')
plt.plot(axisX, sumfp, '-o', label='FP')
plt.plot(axisX, sumaa, '-.', label='AA')
plt.xlabel('N-links')
plt.ylabel('Sum-rates/Mbps')
plt.axis([0, (step-1)*50, 0, max(sumfp[step-1], sumdl[step-1])])
plt.legend()
plt.savefig("0322-2.png")
plt.show()


"""
FP is optimal !
1.When the layouts contain links of similar distances, many distinct local optima emerge, which tend to confuse the
supervised learning process. 
2. It is worth noting that while the channel gains are needed at the training stage for computing rates,
they are not needed for scheduling, which only requires GLI
"""




