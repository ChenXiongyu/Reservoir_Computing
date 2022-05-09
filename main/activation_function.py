import numpy as np


def tanh(array):
    return np.tanh(array)


def relu(array):
    return (abs(array) + array) / 2


def sigmoid(array):
    return 1 / (1 + np.exp(-array))


def prelu(array, alpha=0.01):
    fx = np.zeros(len(array))
    fx[array >= 0] = array[array >= 0]
    fx[array < 0] = alpha * array[array < 0]
    return fx


def elu(array, alpha=1):
    fx = np.zeros(len(array))
    fx[array >= 0] = array[array >= 0]
    fx[array < 0] = alpha * (np.exp(array) - 1)[array < 0]
    return fx


def soft_plus(array):
    return np.log(1 + np.exp(array))
