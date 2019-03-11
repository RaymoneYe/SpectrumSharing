import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio
from itertools import chain

# N = 500
M1 = 5
M2 = 15
M3 = 31
M = 31
p = 1
Gt = 2.5
Gr = 2.5
Ptdbm = 40
fmhz = 2400
NdBm = -169
Noise = (10 ** (NdBm/10))*0.001
w = 1
region = 500
B = 5e6


def linkpair(N):
    t = np.random.random(N) * 2 * np.pi - np.pi
    t = np.expand_dims(t, axis=1)
    x = np.cos(t)
    y = np.sin(t)
    i_set = np.arange(0, N, 1)
    Tx = np.random.uniform(0, region, N)
    Tx = np.expand_dims(Tx, axis=1)     # 变成列矩阵
    Ty = np.random.uniform(0, region, N)
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
    return T, R, len


def hpara(T, R, linknumber):
    h = np.zeros([linknumber, linknumber], dtype=float)
    H = np.zeros([linknumber, linknumber], dtype=float)
    for i in range(0, linknumber):
        for j in range(0, linknumber):
            h[i, j] = 32.45 + 20*math.log10(np.sqrt((T[j, 0] - R[i, 0])**2 + (T[j, 1] - R[i, 1])**2)/1000)\
                      + 20*math.log(fmhz, 10)-Gt-Gr
# 当Pi=1时， H[ii]数值上等于pi*h[i,j]^2,
    Prdbm = Ptdbm - h     # (L=Pt/Pr)
    for i in range(0, linknumber):
        a = Prdbm[i] / 10
        H[i] = (10 ** a) * 0.001
    return H


def iteration(x, y, z, H, Noise, Linknumber):
    for t in range(0, 101):
        for i in range(0, Linknumber):
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


def updateH(H, x, Linknumber):
    H1 = np.copy(H)
    for i in range(0, Linknumber):
        if x[i] == 0:
            H1[i, :] = 0
            H1[:, i] = 0
    return H1


def rates(x, H, Linknumber):
    # return sum-rate of all active links, and the link number
    rate = np.zeros([Linknumber, 1], dtype=float)
    actlinknum = 0
    snr = np.zeros((Linknumber, 1))
    sum1 = 0
    for i in range(0, Linknumber):
        if H[i, i] != 0:
            actlinknum = actlinknum + 1
            for j in range(0, Linknumber):
                sum1 = sum1 + H[i, j]*x[j]
            snr[i] = H[i, i]/(sum1 - H[i, i]*x[i] + Noise)
            sum1 = 0
            rate[i] = 5*np.log2(1 + snr[i])
    return actlinknum, rate


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


def zhouwei(t_density, r_density, Tdensity, Rdensity, M0, M, linknumber):
    # 返回每个发射接收机的周围环境信息（准备做卷积）
    # 分别存储每个链路的发射机/接收机的周围环境（减掉了自身）
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


step = 2
mapx = np.zeros(step)
mapy = np.zeros(step)
mapz = np.zeros(step)
mapk = np.zeros(step)
for kk in range(1, step):
    N = 50*kk
    K = 5000   # 地图数
    actlinknum = np.zeros(K)
    actlinknum1 = np.zeros(K)
    ratess = np.zeros(K)
    ratess1 = np.zeros(K)
    # ============================ #
    Taround1 = np.zeros((K, N, M1, M1))
    Taround2 = np.zeros((K, N, M2, M2))
    Taround3 = np.zeros((K, N, M3, M3))
    Raround1 = np.zeros((K, N, M1, M1))
    Raround2 = np.zeros((K, N, M2, M2))
    Raround3 = np.zeros((K, N, M3, M3))
    tden = np.zeros((K, N, 3))
    rden = np.zeros((K, N, 3))
    Hall = np.zeros((K, N, N))
    Sumrate = 0
    SumrateAA = 0
    Xall = np.zeros((K, N, 1))
    lengthall = np.zeros((K, N, 1))
    # =======main func===========
    for k in range(0, K):
        #  ===随机初始化 =====
        xx = np.random.randint(0, 2, (N, 1))  # [low, high).
        x = np.ones([N, 1], dtype=float)
        x1 = np.ones([N, 1], dtype=float)
        z = np.zeros([N, 1], dtype=float)
        y = np.zeros([N, 1], dtype=float)
        for i in range(0, N):
            if xx[i] == 0:
                x[i] = 0
        [T, R, len] = linkpair(N)
        H = hpara(T, R, N)
        x = iteration(x, y, z, H, Noise, N)
        H0 = updateH(H, x, N)
        H1 = updateH(H, x1, N)
        actlinknum[k], rate = rates(x, H0, N)
        actlinknum1[k], rate1 = rates(x1, H1, N)
        ratess[k] = sum(rate)
        ratess1[k] = sum(rate1)
        # ========== 存储训练数据  =========#
        tdensity, rdensity, Tdensity, Rdensity = density(T, R, M, N)
        Tzhouwei1, Rzhouwei1 = zhouwei(tdensity, rdensity, Tdensity, Rdensity, M1, M, N)
        Tzhouwei2, Rzhouwei2 = zhouwei(tdensity, rdensity, Tdensity, Rdensity, M2, M, N)
        Tzhouwei3, Rzhouwei3 = zhouwei(tdensity, rdensity, Tdensity, Rdensity, M3, M, N)
        tden, rden, Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall = storagedataset(
            k, H, tdensity, rdensity, tden, rden, Tzhouwei1, Rzhouwei1, Tzhouwei2, Rzhouwei2, Tzhouwei3, Rzhouwei3, x,
            len, Taround1, Taround2, Taround3, Raround1, Raround2, Raround3, Xall, lengthall, Hall)
    # ============存储到文件=========== #
    Taround1.tofile('500Tard1.dat', sep=',', format='%f')
    Taround2.tofile('500Tard2.dat', sep=',', format='%f')
    Taround3.tofile('500Tard3.dat', sep=',', format='%f')
    Raround1.tofile('500Rard1.dat', sep=',', format='%f')
    Raround2.tofile('500Rard2.dat', sep=',', format='%f')
    Raround3.tofile('500Rard3.dat', sep=',', format='%f')
    lengthall.tofile('500lenall.dat', sep=',', format='%f')
    # ================================ #
    mapx[kk] = N
    mapy[kk] = sum(actlinknum)/(K*N)*100
    mapz[kk] = np.sum(ratess)/K
    mapk[kk] = np.sum(ratess1)/K
    print('N =', N)
    print('Aver-activate rate:', sum(actlinknum)/(K*N)*100, '%')
    print('Aver-sumrate:', np.sum(ratess)/K, 'Mbps')
    print('All-Active-Aver-sumrate:', np.sum(ratess1)/K, 'Mbps')
"""    
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
plt.axis([100, 1600, 0, 3000])
plt.legend()
# plt.savefig('name')
plt.show()
"""

