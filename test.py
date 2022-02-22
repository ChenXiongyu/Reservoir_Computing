import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 3

W_r = np.random.random((N, N))
W_in = np.random.random((N, D))

Gamma = 10
Sigma = 0.4
Rou_in = 0.3
Rou_r = 0.3
K = 1

hyper_parameters = {"gamma": Gamma,
                    "sigma": Sigma,
                    "rou_in": Rou_in,
                    "k": K,
                    "rou_r": Rou_r}


def w_in_construction(sigma, rou_in, n, d):

    w_in = np.random.randn(n, d) * rou_in
    index = np.array(np.random.uniform(0, 1, (n, d)) < sigma, dtype=int)

    return index * w_in


def out_layer(w_out, f_out, r):

    length = r.shape[1]
    y = np.zeros(r.shape)

    for t in range(length):
        y[:, t] = np.dot(w_out, f_out(r[:, t]))

    return y


def forcasting(gamma, w_r, w_in, w_out, f_out, r_0, delta_t, forcast_time):

    dimension = len(r_0)
    r = np.zeros((dimension, forcast_time + 1))
    r[:, 0] = r_0

    for t in range(forcast_time):

        derivative = -gamma * r[:, t] + gamma * np.tanh(np.dot(w_r, r[:, t]) +
                                                        np.dot(w_in, np.dot(w_out, f_out(r[:, t]))))
        r[:, t + 1] = r[:, t] + derivative * delta_t

    return r
