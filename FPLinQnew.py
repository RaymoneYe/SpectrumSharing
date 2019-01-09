import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import chain

N = 100
p = 1
sigma = 0
Gt = 2.5
Gr = 2.5
Ptdbm = 40
fmhz = 2400
NdBm = -169
N = (10 ** (NdBm/10))*0.001
w = 1
xx = 0
yy = 0
zz = 0
xp = 0

trans = np.empty([N, 2], dtype=int)
recv = np.empty([N, 2], dtype=int)
dist = np.empty([N, 1], dtype=int)
h = np.empty([N, N], dtype=float)
H = np.empty([N, N], dtype=float)
for i in range(0, N):
    trans[i] = np.random.randint(0, 1000, size=[1, 2])
    dist[i] = np.random.uniform(2, 65)
    recv[i, 0] = np.random.randint(max(0, trans[i, 0] - dist[i]), min(1000, trans[i, 0] + dist[i]), size=[1, 1])
    recv[i, 1] = np.random.randint(max(0, trans[i, 1] - dist[i]), min(1000, trans[i, 1] + dist[i]), size=[1, 1])
    while 1:
        dis = int(np.sqrt(np.sum(np.square(trans[i] - recv[i]))))
        if dis == dist[i]:
            break
        else:
            recv[i, 0] = np.random.randint(max(0, trans[i, 0] - dist[i]), min(1000, trans[i, 0] + dist[i]), size=[1, 1])
            recv[i, 1] = np.random.randint(max(0, trans[i, 1] - dist[i]), min(1000, trans[i, 1] + dist[i]), size=[1, 1])

for i in range(0, N):
    plt.plot([trans[i, 0], recv[i, 0]], [trans[i, 1], recv[i, 1]], '-b')
    # plt.scatter(trans[2, 0], trans[2, 1], c='r', marker='*')
    # plt.scatter(recv[2, :], '.b')

plt.plot([trans[2, 0], recv[2, 0]], [trans[2, 1], recv[2, 1]], '-r')
plt.show()
print('trans[10]:', trans[10], '\n recv[5]:', recv[5], '\n dist[5]:', dist[5], '\nk:',
      int(np.sqrt(np.sum(np.square(trans[10] - recv[5])))))

for i in range(0, N):
    for j in range(0, N):
        h[i, j] = 32.45 + 20*math.log10(np.sqrt(np.sum(np.square(trans[j] - recv[i])))/1000)+20*math.log10(fmhz)-Gt-Gr

Prdbm = Ptdbm - h
for i in range(0, N):
    a = Prdbm[i] / 10
    H[i] = (10 ** a) * 0.001


print('h[5,10]=%.4f'%h[5, 10])
print('h[5,5]=%.4f'%h[10, 10])
# One D2D net created


x = np.ones([N, 1], dtype=float)
xx = np.empty([N, 1], dtype=float)
z = np.empty([N, 1], dtype=float)
y = np.empty([N, 1], dtype=float)

for t in range(0, 20):
    for i in range(0, N):
        for j in chain(range(0, i), range(i+1, N)):
            zz = zz + h[i, j] * x[j] * p
        z[i] = (h[i, i] * p * x[i])/(zz + sigma)
        zz = 0
        for j in range(0, N):
            yy = yy + h[i, j]*p*x[j]
        y[i] = (math.sqrt(w*(1+z[i])*h[i, i]*p*x[i]))/(yy + sigma)
        yy = 0
        for j in range(0, N):
            xp = xp + y[i] *h[j, i] * p
        xx[i] = pow((y[i] * math.sqrt(w * (1 + z[i]) * h[i, i] * p)) / xp, 2)
        x[i] = min(1, xx[i])
        Q = 2*y[i]*math.sqrt(w*(1+z[i])*h[i, i]*x[i]) - pow(xp*math.sqrt(x[i]), 2)
        if Q > 0:
            x[i] = 1
        else:
            x[i] = 0
        xp = 0

print('z[45]=%.10f:'%z[45])
print('y[45]=%.10f'%y[45])
print('xx[45]=%.10f'%xx[45])
print('x[45]=%.10f'%x[45])
# print(x[0:10])

