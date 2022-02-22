import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 3

W_r = np.random.random((N, N))
W_in = np.random.random((N, D))

Gamma = 10


def reservoir(gamma, w_r, w_in, u, r_0, delta_t):

    dimension = u.shape[0]
    length = u.shape[1]

    r = np.zeros((dimension, length + 1))
    r[:, 0] = r_0.T

    for t in range(length):
        derivative = -gamma * r[:, t] + gamma * np.tanh(np.dot(w_r, r[:, t]) +
                                                        np.dot(w_in, u[:, t]))
        r[:, t + 1] = r[:, t] + derivative * delta_t

    return r


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


def lorenz_63(start_pos, time_tick, delta_t):

    trajectory = np.zeros((len(start_pos), time_tick + 1))
    trajectory[:, 0] = start_pos

    for t in range(time_tick):
        derivative = np.array([10 * (trajectory[1, t] - trajectory[0, t]),
                               trajectory[0, t] * (28 - trajectory[2, t]) - trajectory[1, t],
                               trajectory[0, t] * trajectory[1, t] - 8 / 3 * trajectory[2, t]])
        trajectory[:, t + 1] = trajectory[:, t] + derivative * delta_t

    return trajectory


def plot_3_dimension(trajectory, save_path=''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :])
    if save_path:
        plt.savefig(save_path)
        plt.close()

