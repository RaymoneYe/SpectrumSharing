import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import chain

N = 300
p = 1
Gt = 2.5
Gr = 2.5
Ptdbm = 40
fmhz = 2400
NdBm = -184
Noise = (10 ** (NdBm/10))*0.001
w = 1
region = 1000
B = 5e6


h = np.empty([N, N], dtype=float)
H = np.empty([N, N], dtype=float)
rate = np.zeros([N, 1], dtype=float)


def linkpair(N):
    t = np.random.random(N) * 2 * np.pi - np.pi
    t = np.expand_dims(t, axis=1)
    x = np.cos(t)
    y = np.sin(t)
    i_set = np.arange(0, N, 1)
    Tx = np.random.uniform(0, 1000, N)
    Tx = np.expand_dims(Tx, axis=1)
    Ty = np.random.uniform(0, 1000, N)
    Ty = np.expand_dims(Ty, axis=1)
    len = np.zeros((N, 1))
    for i in i_set:
        len[i] = np.random.uniform(2, 65)
        x[i] = x[i] * len[i]
        y[i] = y[i] * len[i]
        Rx = Tx + x
        Ry = Ty + y
        for i in range(0, N):
            if Rx[i] > region or Rx[i] < 0:
                Rx[i] = Tx[i] - 2 * x[i]
            if Ry[i] > region or Ry[i] < 0:
                Ry[i] = Ty[i] - 2 * y[i]
        T = np.vstack((Tx.T, Ty.T)).T
        R = np.vstack((Rx.T, Ry.T)).T
    return T, R


[T, R] = linkpair(N)
for i in range(0, N):
    plt.plot([T[i, 0], R[i, 0]], [T[i, 1], R[i, 1]], '-r')
    plt.axis([0, 1010, 0, 1010])


for i in range(0, N):
    for j in range(0, N):
        h[i, j] = 32.45 + 20*math.log10(np.sqrt((T[j, 0] - R[i, 0])**2 + (T[j, 1] - R[i, 1])**2)/1000)\
                  + 20*math.log(fmhz, 10)-Gt-Gr
# 当Pi=1时， H[ii]数值上等于pi*h[i,j]^2,
Prdbm = Ptdbm - h     # (L=Pt/Pr)
for i in range(0, N):
    a = Prdbm[i] / 10
    H[i] = (10 ** a) * 0.001


print('H[5,10]=%.14f' % H[5, 10])
print('H[5,5]=%.14f W' % H[10, 10])
# One D2D net created


xx = np.random.randint(0, 2, (N, 1))  # [low, high).
x = np.ones([N, 1], dtype=float)
z = np.empty([N, 1], dtype=float)
y = np.empty([N, 1], dtype=float)
for i in range(0, N):                 # 随机初始化
    if xx[i] == 0:
        x[i] = 0


for t in range(0, 101):
    for i in range(0, N):
        z[i] = (H[i, i] * x[i])/(np.dot(H[i, :].T, x) - H[i, i]*x[i] + Noise)   # numpyarray.T: 转置
        y[i] = math.sqrt(w*(1+z[i])*H[i, i]*p*x[i])/(np.dot(H[i, :].T, x) + Noise)
        xxx = (y[i] * np.sqrt(w * (1 + z[i]) * H[i, i] * p)) / np.dot(H[:, i].T, np.multiply(y, y))
        # xxx = y[i] * np.sqrt(w * (1 + z[i]) * H[i, i]) / (np.dot(H[i, :].T, np.multiply(y, y)))
        x[i] = min(1, xxx*xxx)
for i in range(0, N):
    Q = 2*y[i]*math.sqrt(w*(1+z[i])*H[i, i]) - (np.dot(H[:, i].T, np.multiply(y, y)))
    if Q > 0:
        x[i] = 1
    else:
        x[i] = 0

counter = 0
for i in range(0, N):
    if x[i] == 0:
        H[i, :] = 0
        H[:, i] = 0
snr = np.zeros((N, 1))
sum1 = 0
for i in range(0, N):
    if H[i, i] != 0:
        for j in range(0, N):
            sum1 = sum1 + H[i, j]*x[j]
        snr[i] = H[i, i]/(sum1 - H[i, i]*x[i] + Noise)
        sum1 = 0
        rate[i] = 5*np.log2(1 + snr[i])
        counter = counter + 1
        plt.plot([T[i, 0], R[i, 0]], [T[i, 1], R[i, 1]], '-b')
        plt.axis([0, 1050, 0, 1050])
print('Activate rate:', counter/N*100, '%')
print('sumrate:', np.sum(rate), 'Mbps')
plt.show()
