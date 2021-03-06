import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import chain

# N = 500
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


def hpara(T, R, N):
    h = np.zeros([N, N], dtype=float)
    H = np.zeros([N, N], dtype=float)
    for i in range(0, N):
        for j in range(0, N):
            h[i, j] = 32.45 + 20*math.log10(np.sqrt((T[j, 0] - R[i, 0])**2 + (T[j, 1] - R[i, 1])**2)/1000)\
                      + 20*math.log(fmhz, 10)-Gt-Gr
# 当Pi=1时， H[ii]数值上等于pi*h[i,j]^2,
    Prdbm = Ptdbm - h     # (L=Pt/Pr)
    for i in range(0, N):
        a = Prdbm[i] / 10
        H[i] = (10 ** a) * 0.001
    return H


def iteration(x, y, z, H, Noise):
    for t in range(0, 101):
        for i in range(0, N):
            z[i] = (H[i, i] * x[i])/(np.dot(H[i, :].T, x) - H[i, i]*x[i] + Noise)   # numpyarray.T: 转置
            y[i] = math.sqrt(w*(1+z[i])*H[i, i]*p*x[i])/(np.dot(H[i, :].T, x) + Noise)
            xxx = (y[i] * np.sqrt(w * (1 + z[i]) * H[i, i] * p)) / (np.dot(H[:, i].T, (np.multiply(y, y))))
            x[i] = min(1, xxx*xxx)
    for i in range(0, N):
        Q = 2*y[i]*math.sqrt(w*(1+z[i])*H[i, i]) - (np.dot(H[:, i].T, np.multiply(y, y)))
        if Q > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x


def updateH(H, x):
    H1 = np.copy(H)
    for i in range(0, N):
        if x[i] == 0:
            H1[i, :] = 0
            H1[:, i] = 0
    return H1


def rates(x, H):
    # return sum-rate of all active links, and the link number
    rate = np.zeros([N, 1], dtype=float)
    actlinknum = 0
    snr = np.zeros((N, 1))
    sum1 = 0
    for i in range(0, N):
        if H[i, i] != 0:
            actlinknum = actlinknum + 1
            for j in range(0, N):
                sum1 = sum1 + H[i, j]*x[j]
            snr[i] = H[i, i]/(sum1 - H[i, i]*x[i] + Noise)
            sum1 = 0
            rate[i] = 5*np.log2(1 + snr[i])
            # plt.plot([T[i, 0], R[i, 0]], [T[i, 1], R[i, 1]], '-b')
            # plt.axis([0, 1050, 0, 1050])
#           print(counter)
    return actlinknum, rate


step = 11
mapx = np.zeros(step)
mapy = np.zeros(step)
mapz = np.zeros(step)
mapk = np.zeros(step)
for kk in range(1, step):
    N = 100*kk
    K = 50
    actlinknum = np.zeros(K)
    actlinknum1 = np.zeros(K)
    ratess = np.zeros(K)
    ratess1 = np.zeros(K)
    for k in range(0, K):
        # ===随机初始化 =====
        xx = np.random.randint(0, 2, (N, 1))  # [low, high).
        x = np.ones([N, 1], dtype=float)
        x1 = np.ones([N, 1], dtype=float)
        z = np.zeros([N, 1], dtype=float)
        y = np.zeros([N, 1], dtype=float)
        for i in range(0, N):
            if xx[i] == 0:
                x[i] = 0
        # =======main func===========
        [T, R] = linkpair(N)
        H = hpara(T, R, N)
        x = iteration(x, y, z, H, Noise)
        H0 = updateH(H, x)
        H1 = updateH(H, x1)
        actlinknum[k], rate = rates(x, H0)
        actlinknum1[k], rate1 = rates(x1, H1)
        ratess[k] = sum(rate)
        ratess1[k] = sum(rate1)
    mapx[kk] = N
    mapy[kk] = sum(actlinknum)/(K*N)*100
    mapz[kk] = np.sum(ratess)/K
    mapk[kk] = np.sum(ratess1)/K
    print('N =', N)
    print('Aver-activate rate:', sum(actlinknum)/(K*N)*100, '%')
    print('Aver-sumrate:', np.sum(ratess)/K, 'Mbps')
    print('All-Active-Aver-sumrate:', np.sum(ratess1)/K, 'Mbps')
print(mapx)
print(mapy)
print(mapz)
print(mapk)
plt.figure(1)
plt.plot(mapx, mapy, '-*')
plt.axis([100, 1000, 0, 50])
plt.xlabel('N-links')
plt.ylabel('Activa-Rate: %')
plt.legend()  # 展示图例
plt.figure(2)
plt.plot(mapx, mapz, '-*', label='FP')
plt.plot(mapx, mapk, '-o', label='All-Active')
plt.xlabel('N-links')
plt.ylabel('Sum-rates/Mbps')
plt.axis([0, 1000, 0, 3000])
plt.legend()
plt.show()
