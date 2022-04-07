import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def diff_lorenz(u):
    sigma = 10
    r = 28
    b = 8 / 3
    x, y, z = u
    f = [sigma * (y - x), r * x - y - x * z, x * y - b * z]
    df = [[-sigma, sigma, 0], [r - z, -1, -x], [y, x, -b]]
    return np.array(f), np.array(df)


def diff_rossler(u):
    x, y, z = u
    a = 0.2
    b = 0.2
    c = 5.7
    f = [- y - z, x + a * y, b + z * (x - c)]
    df = [[0, -1, -1], [1, a, 0], [z, 0, x]]
    return np.array(f), np.array(df)


def lec_system(t):
    # x,y,z = Max_t[:3]             # n=3
    u = t[3:12].reshape([3, 3])  # size n square matrix, sub-array from n to n+n*n=n*(n+1)
    # l = Max_t[12:15]  # vector, sub-array from n*(n+1) to n*(n+1)+n=n*(n+2)
    f, df = diff_rossler(t[:3])
    a = u.T.dot(df.dot(u))
    dl = np.diag(a).copy()
    for i in range(3):
        a[i, i] = 0
        for j in range(i + 1, 3):
            a[i, j] = -a[j, i]
    du = u.dot(a)
    return np.concatenate([f, du.flatten(), dl])


u0 = np.ones(3)
U0 = np.identity(3)
L0 = np.zeros(3)
u0 = np.concatenate([u0, U0.flatten(), L0])
T = np.linspace(0, 1000, 10001)

U = odeint(lambda u, t: lec_system(u), u0, T)
L = U[5:, 12:15].T / T[5:]

plt.plot(T[5:], L.T)
print(L[:, -1])
