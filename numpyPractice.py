import numpy as np
import matplotlib.pyplot as plt


N = 1000
trans = np.empty([N, 2], dtype=int)
recv = np.empty([N, 2], dtype=int)
dist = np.empty([N, 1], dtype=int)

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

k = int(np.sqrt(np.sum(np.square(trans[2] - recv[2]))))
print('trans[2]:', trans[2], '\n recv[2]:', recv[2], '\n dist[2]:', dist[2], '\nk:', k)

for i in range(0, N):
    plt.plot([trans[i, 0], recv[i, 0]], [trans[i, 1], recv[i, 1]], '-b')
    # plt.scatter(trans[2, 0], trans[2, 1], c='r', marker='*')
    # plt.scatter(recv[2, :], '.b')

plt.plot([trans[2, 0], recv[2, 0]], [trans[2, 1], recv[2, 1]], '-r')
plt.show()

