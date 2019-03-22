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
    # 产生信道参数H矩阵，采用对数衰落模型。return H数值上等于
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


def iteration(x, y, z, H, Noise, linknumber):
    g = 100
    # 迭代g次直到收敛
    for t in range(0, g):
        for i in range(0, linknumber):
            z[i] = (H[i, i] * x[i])/(np.dot(H[i, :].T, x) - H[i, i]*x[i] + Noise)   # numpyarray.T: 转置
            y[i] = math.sqrt(w*(1+z[i])*H[i, i]*p*x[i])/(np.dot(H[i, :].T, x) + Noise)
            xxx = (y[i] * np.sqrt(w * (1 + z[i]) * H[i, i] * p)) / (np.dot(H[:, i].T, (np.multiply(y, y))))
            x[i] = min(1, xxx*xxx)
    for i in range(0, linknumber):
        Q = 2*y[i]*math.sqrt(w*(1+z[i])*H[i, i]) - (np.dot(H[:, i].T, np.multiply(y, y)))
        if Q > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x


def iteration3(x, y, z, H, Noise, linknumber):
    g = 100
    root = np.array([0, 0.5, 1])
    for t in range(0, g):
        for i in range(0, linknumber):
            z[i] = (H[i, i] * x[i])/(np.dot(H[i, :].T, x) - H[i, i]*x[i] + Noise)   # numpyarray.T: 转置
            y[i] = math.sqrt(w*(1+z[i])*H[i, i]*p*x[i])/(np.dot(H[i, :].T, x) + Noise)
            xxx = (y[i] * np.sqrt(w * (1 + z[i]) * H[i, i] * p)) / (np.dot(H[:, i].T, (np.multiply(y, y))))
            x[i] = min(1, xxx*xxx)
    for i in range(0, linknumber):
        xxx = (y[i] * np.sqrt(w * (1 + z[i]) * H[i, i] * p)) / (np.dot(H[:, i].T, (np.multiply(y, y))))
        x[i] = root[np.argmin(np.abs(root-xxx*xxx))]
    return x


def updateH(H, x, linknumber):
    H1 = np.copy(H)
    for i in range(0, linknumber):
        if x[i] == 0:
            H1[i, :] = 0
            H1[:, i] = 0
    return H1


def rates(x, H, linknumber):
    # return sum-rate of all active links, and the link number
    rate = np.zeros([linknumber, 1], dtype=float)
    actlinknum = 0
    snr = np.zeros((linknumber, 1))
    sum1 = 0
    for i in range(0, linknumber):
        if H[i, i] != 0:
            actlinknum = actlinknum + 1
            for j in range(0, linknumber):
                sum1 = sum1 + H[i, j]*x[j]
            snr[i] = H[i, i]/(sum1 - H[i, i]*x[i] + Noise)
            sum1 = 0
            rate[i] = 5*np.log2(1 + snr[i])
    return actlinknum, rate


def sortlen(len, aclinknum, linknumber):
    lenc = np.copy(len)
    aclink = np.array([21, 31, 39, 47, 53, 60, 65, 70, 77, 82])  # 只用于链路数量为100的整数倍的情况下
    lensorted = np.zeros((linknumber, 1))
    xs = np.zeros((linknumber, 1), dtype=float)
    lensorted = quick_sort(lenc, 1, linknumber-1)   # 0号最小
    # 下面是FP5写法, 只能用于100的整数倍link数量

    for j in range(1, linknumber):
        if len[j] <= lensorted[aclink[int(linknumber/100)]]:
            xs[j] = 1
        if len[j] > lensorted[aclink[int(linknumber/100)]]:
            xs[j] = 0
    """
    # 下面是FP2写法
    for j in range(1, linknumber):
        if len[j] <= lensorted[int(aclinknum)]:
            xs[j] = 1
        if len[j] > lensorted[int(aclinknum)]:
            xs[j] = 0
    """
    return xs


def quick_sort(lists, left, right):
    # 快速排序
    if left >= right:
        return lists
    key = lists[left]
    low = left
    high = right
    while left < right:
        while left < right and lists[right] >= key:
            right -= 1
        lists[left] = lists[right]
        while left < right and lists[left] <= key:
            left += 1
        lists[right] = lists[left]
    lists[right] = key
    quick_sort(lists, low, left - 1)
    quick_sort(lists, left + 1, high)
    return lists


step = 5
mapx = np.zeros(step)
mapy = np.zeros(step)
mapz = np.zeros(step)
mapk = np.zeros(step)
mapl = np.zeros(step)
mapr = np.zeros(step)
for kk in range(1, step):
    N = 50*kk
    MAPS = 100
    actlinknum = np.zeros(MAPS)
    actlinknum1 = np.zeros(MAPS)
    actlinknum2 = np.zeros(MAPS)
    actlinknum3 = np.zeros(MAPS)
    ratess = np.zeros(MAPS)
    ratess1 = np.zeros(MAPS)
    ratess2 = np.zeros(MAPS)
    ratess3 = np.zeros(MAPS)
    for k in range(0, MAPS):
        #  ===随机初始化 =====
        xx = np.random.randint(0, 2, (N, 1))  # [low, high).
        x = np.ones([N, 1], dtype=float)
        x1 = np.ones([N, 1], dtype=float)
        z = np.zeros([N, 1], dtype=float)
        y = np.zeros([N, 1], dtype=float)
        xs = np.zeros([N, 1], dtype=float)
        xr = np.random.randint(0, 2, [N, 1])
        """
         for i in range(0, N):
            if xx[i] == 0:
                x[i] = 0
        """
        # =======main func===========
        [T, R, length] = linkpair(N)
        H = hpara(T, R, N)
        x = iteration(x, y, z, H, Noise, N)
        H0 = updateH(H, x, N)   # FP
        H1 = updateH(H, x1, N)  # AA
        actlinknum[k], rate = rates(x, H0, N)      # FP
        xs = sortlen(length, actlinknum[k], N)     # SLF
        H2 = updateH(H, xs, N)     # ShortLinkFirst
        H3 = updateH(H, xr, N)     # Random Active
        actlinknum2[k], rate2 = rates(xs, H2, N)   # ShortLinkFirst
        actlinknum3[k], rate3 = rates(xr, H3, N)   # Random
        actlinknum1[k], rate1 = rates(x1, H1, N)   # All-Active
        ratess[k] = sum(rate)
        ratess1[k] = sum(rate1)
        ratess2[k] = sum(rate2)
        ratess3[k] = sum(rate3)
    mapx[kk] = N
    mapy[kk] = sum(actlinknum)/(MAPS*N)*100
    mapz[kk] = np.sum(ratess)/MAPS
    mapk[kk] = np.sum(ratess1)/MAPS
    mapl[kk] = np.sum(ratess2)/MAPS
    mapr[kk] = np.sum(ratess3)/MAPS
    print('N =', N)
    print('Aver-activate rate:', sum(actlinknum)/(MAPS*N)*100, '%')
    print('FP:', np.sum(ratess)/MAPS, 'Mbps')
    print('AA:', np.sum(ratess1)/MAPS, 'Mbps')
    print('SLF', np.sum(ratess2)/MAPS, 'Mbps')
    print('Random', np.sum(ratess3)/MAPS, 'Mbps')
print(mapx)
print(mapy)
print(mapz)
print(mapk)
print(mapl)
print(mapr)
plt.figure(1)
plt.plot(mapx, mapy, '-*')
plt.axis([100, 100*(step-1), 0, 30])
plt.xlabel('N-links')
plt.ylabel('Activa-Rate: %')
plt.legend()  # 展示图例
plt.figure(2)
plt.plot(mapx, mapz, '-*', label='FP')
plt.plot(mapx, mapk, '-o', label='All-Active')
plt.plot(mapx, mapl, '-.', label='ShortFirst')
plt.plot(mapx, mapr, '-+', label='Random')
plt.xlabel('N-links')
plt.ylabel('Sum-rates/Mbps')
plt.axis([100, 100*(step-1), 0, 1500])
plt.legend()
plt.show()
