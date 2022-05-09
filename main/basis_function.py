import numpy as np


def original(array):
    return array


def square(array):
    return np.square(array)


def sin(array, k=1):
    return np.sin(k * array)


def cos(array, k=1):
    return np.cos(k * array)
