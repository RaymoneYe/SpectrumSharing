# -*- coding: utf-8 -*-
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


# $$$$$$$$$ 初始化  $$$$$$$$$$$
region = 500
Ptdbm = 40
fmhz = 2400
Gt = 2.5
Gr = 2.5
Ndbm = -184
N = (10 ** (Ndbm / 10)) * 0.001
B = 5e6
M1 = 5
M2 = 15
M3 = 31
M = 31
w =1
p = 1
internumber = 50   # 地图数/组数
testinternumber = 50   # 验证集数量


def delete(X, H, testnumber):
    H1 = np.copy(H)
    for i in range(0, testnumber):
        if X[i] == 0:
            H1[i, :] = 0
            H1[:, i] = 0
    return H1


def rate(H, N, B, testnumber):
    rate = np.zeros((testnumber, 1))
    snr = np.zeros((testnumber, 1))
    for i in range(0, testnumber):
        if H[i, i] != 0:
            hi = H[i, :]
            snr[i] = H[i, i] / (np.sum(hi) - H[i, i] + N)
            rate[i] = B * math.log(snr[i] + 1, 2)
    return rate


def creat_model(reg):
    it1 = Input(shape=(M1, M1, 1), name='xd1')
    it2 = Input(shape=(M2, M2, 1), name='xd2')
    it3 = Input(shape=(M3, M3, 1), name='xd3')
    it4 = Input(shape=(M1, M1, 1), name='yd1')
    it5 = Input(shape=(M2, M2, 1), name='yd2')
    it6 = Input(shape=(M3, M3, 1), name='yd3')  # 输入
    it7 = Input(shape=(1,), name='lenth')  # 输入1维，不指定个数

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
    out_0 = Dense(21, activation='relu', W_regularizer=l2(reg))(x)  # regualrizer正则项
    out_1 = Dense(21, activation='relu', W_regularizer=l2(reg))(out_0)
    out_2 = Dense(1, activation='sigmoid', W_regularizer=l2(reg))(out_1)
    model = Model(inputs=[it1, it2, it3, it4, it5, it6, it7], outputs=[out_2])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


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
    model.save_weights('temple2.h5')
    model.load_weights('temple2.h5')
    #  predict函数按batch获得输入数据对应的输出
    xt = model.predict(
        [np.expand_dims(Tzw1, axis=-1), np.expand_dims(Tzw2, axis=-1),
         np.expand_dims(Tzw3, axis=-1), np.expand_dims(Rzw1, axis=-1),
         np.expand_dims(Rzw2, axis=-1), np.expand_dims(Rzw3, axis=-1),
         length], batch_size=1)  # axis=-1表示最后一维被扩展一维
    # return continuous value
    return xt, model


#   $$$$$$$  训练网络  $$$$$$$   #
reg = 0.01
step = 2
xn = 1
sumfp = np.zeros(step)
sumaa = np.zeros(step)
sumdl = np.zeros(step)
xstep = np.zeros(step)
model = creat_model(reg)
model.load_weights('temple2.h5')
testnumber = 50
if 2 > 1:
    SumrateDL = 0
    SumrateFP = 0
    Taround1t = np.fromfile('50Tard1.dat', dtype=np.float, sep=',')
    Taround2t = np.fromfile('50Tard2.dat', dtype=np.float, sep=',')
    Taround3t = np.fromfile('50Tard3.dat', dtype=np.float, sep=',')
    Raround1t = np.fromfile('50Rard1.dat', dtype=np.float, sep=',')
    Raround2t = np.fromfile('50Rard2.dat', dtype=np.float, sep=',')
    Raround3t = np.fromfile('50Rard3.dat', dtype=np.float, sep=',')
    lengthallt = np.fromfile('50lenall.dat', dtype=np.float, sep=',')
    Hallt = np.fromfile('50Hall.dat', dtype=np.float, sep=',')
    Xallt = np.fromfile('50Xall.dat', dtype=np.float, sep=',')
    Taround1t = np.reshape(Taround1t, (50, 50, 5, 5))
    Taround2t = np.reshape(Taround2t, (50, 50, 15, 15))
    Taround3t = np.reshape(Taround3t, (50, 50, 31, 31))
    Raround1t = np.reshape(Raround1t, (50, 50, 5, 5))
    Raround2t = np.reshape(Raround2t, (50, 50, 15, 15))
    Raround3t = np.reshape(Raround3t, (50, 50, 31, 31))
    lengthallt = np.reshape(lengthallt, (50, 50, 1))
    Hallt = np.reshape(Hallt, (50, 50, 50))
    Xallt = np.reshape(Xallt, (50, 50, 1))
    for i in range(0, testnumber):  # testinternumber = 10, testnumber = 50
        model.load_weights('goodmodel.h5')
        #  predict函数按batch获得输入数据对应的输出
        res = model.predict(
            [np.expand_dims(Taround1t[i, :, :, :], axis=-1), np.expand_dims(Taround2t[i, :, :, :], axis=-1),
             np.expand_dims(Taround3t[i, :, :, :], axis=-1), np.expand_dims(Raround1t[i, :, :, :], axis=-1),
             np.expand_dims(Raround2t[i, :, :, :], axis=-1), np.expand_dims(Raround3t[i, :, :, :], axis=-1),
             lengthallt[i, :, :]], batch_size=1)  # axis=-1表示最后一维被扩展一维
        #    out[i, :, :] = res
        predict = (res > 0.5)  # return True or False
        predict = (predict + 0)  # turn Bool into Int
        Hdl = delete(predict, Hallt[i, :, :], testnumber)
        Hfp = delete(Xallt[i, :, :], Hallt[i, :, :], testnumber)
        RateDL = np.zeros((testnumber, 1))
        RateFP = np.zeros((testnumber, 1))
        RateDL = rate(Hdl, N, B, testnumber)
        RateFP = rate(Hfp, N, B, testnumber)
        print(np.sum(RateFP)/1e6)
        print(np.sum(RateDL)/1e6)
        print('==============')
        SumrateDL = SumrateDL + np.sum(RateDL)/1e6
        SumrateFP = SumrateFP + np.sum(RateFP)/1e6
    # $$$ output
    xstep[xn] = testnumber
    sumfp[xn] = SumrateFP / testinternumber
    sumdl[xn] = SumrateDL / testinternumber
    print('N = ', xn)
    print('test-number = ', testnumber)
    print('FP-sum-rate= ', sumfp[xn], 'MBps')
    print('All-Active sum-rate1= ', sumaa[xn], 'MBps')
    print('DL-sum-rate1= ', sumdl[xn], 'MBps')
"""
plt.figure(1)
plt.plot(xstep, sumfp, '-*', label='FP')
plt.plot(xstep, sumaa, '-o', label='All-Active')
plt.plot(xstep, sumdl, '--', label='SDL')
plt.xlabel('N-links')
plt.ylabel('Sum-rates/Mbps')
plt.axis([50, 1100, 0, 1000])
plt.legend()
plt.show()
"""






