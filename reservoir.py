import numpy as np
from tqdm import tqdm


def initial_reservoir(n_1, n_2, random_type, low=0.0, high=0.0):
    reservoir = np.zeros((n_1, n_2))
    if random_type == 'uniform':
        reservoir = np.random.uniform(low, high, (n_1, n_2))
    if random_type == 'normal':
        reservoir = np.random.randn(n_1, n_2)
    return reservoir


def reservoir_construction_fix_degree(n_1, n_2, random_type, degree, sr=0.0, scale=0.0, low=0.0, high=0.0):

    reservoir = initial_reservoir(n_1, n_2, random_type, low=low, high=high)
    for row in range(n_1):
        index = np.random.choice(n_2, n_2 - degree, replace=False)
        reservoir[row, :][index] = 0

    if sr:
        sr = sr / max(abs(np.linalg.eigvals(reservoir)))
        reservoir = sr * reservoir

    if scale:
        reservoir = scale * reservoir

    return reservoir


def reservoir_construction_average_allocate(n_1, n_2, random_type, low=0.0, high=0.0):

    reservoir_initial = initial_reservoir(n_1, n_2, random_type, low=low, high=high)
    reservoir = np.zeros(reservoir_initial.shape)
    average_node = int(n_1 / n_2)

    index_array = np.ones(n_1)
    for column in range(n_2 - 1):
        index = np.where(index_array == 1)[0]
        index = np.random.choice(index, average_node, replace=False)
        index_array[index] = 0
        reservoir[index, column] = reservoir_initial[index, column]
    index = np.where(index_array == 1)[0]
    reservoir[index, -1] = reservoir_initial[index, -1]

    return reservoir


def reservoir_training(w_r, w_i, reservoir_start, trajectory):
    reservoir_state = np.zeros((len(trajectory), len(reservoir_start)))
    reservoir_state[0, :] = reservoir_start

    for i in tqdm(range(1, len(trajectory))):
        reservoir_state[i, :] = np.tanh(np.dot(w_r, reservoir_state[i - 1, :]) + np.dot(w_i, trajectory[i - 1, :]))

    return reservoir_state


def output_training(reservoir_state, trajectory, beta):
    s = reservoir_state.copy()
    s[:, 1::-1] = s[:, 1::-1] ** 2
    w_0 = np.linalg.solve(np.dot(s.T, s) + beta * np.eye(s.shape[1]), np.dot(s.T, trajectory))
    w_0 = w_0.T

    w_01 = np.zeros(w_0.shape)
    w_02 = np.zeros(w_0.shape)

    w_01[:, ::1] = w_0[:, ::1]
    w_02[:, 1::1] = w_0[:, 1::1]

    output = np.dot(w_01, reservoir_state.T) + np.dot(w_02, reservoir_state.T ** 2)

    def f_0(r):
        return np.dot(w_01, r.T) + np.dot(w_02, r.T ** 2)

    return output.T, f_0


def output_predicting(reservoir_start, trajectory_start, w_r, w_i, f_0, predicting_length):
    reservoir_state = np.zeros((predicting_length, len(reservoir_start)))
    trajectory_predicting = np.zeros((predicting_length, len(trajectory_start)))
    reservoir_state[0, :] = reservoir_start
    trajectory_predicting[0, :] = trajectory_start

    for i in tqdm(range(1, predicting_length)):
        reservoir_state[i, :] = np.tanh(np.dot(w_r, reservoir_state[i - 1, :]) +
                                        np.dot(w_i, trajectory_predicting[i - 1, :]))
        trajectory_predicting[i, :] = f_0(reservoir_state[i, :])

    return trajectory_predicting
