import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def lorenz_system(start_pos, trajectory_length, delta_t):

    trajectory = np.zeros((trajectory_length, len(start_pos)))
    trajectory[0, :] = start_pos

    for t in range(trajectory_length - 1):
        derivative = np.array([10 * (trajectory[t, 1] - trajectory[t, 0]),
                               trajectory[t, 0] * (28 - 10 * trajectory[t, 2]) - trajectory[t, 1],
                               trajectory[t, 0] * trajectory[t, 1] - trajectory[t, 2]])
        trajectory[t + 1, :] = trajectory[t, :] + derivative * delta_t

    return trajectory


def rossler_system(start_pos, trajectory_length, delta_t):

    trajectory = np.zeros((trajectory_length, len(start_pos)))
    trajectory[0, :] = start_pos

    for t in range(trajectory_length - 1):
        derivative = np.array([- trajectory[t, 1] - trajectory[t, 2],
                               trajectory[t, 0] + 0.5 * trajectory[t, 1],
                               2 + trajectory[t, 2] * (trajectory[t, 0] - 4)])
        trajectory[t + 1, :] = trajectory[t, :] + derivative * delta_t

    return trajectory


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


def plot_trajectory(trajectory_1, trajectory_2=np.array([]), save_path=''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(trajectory_1[:, 0], trajectory_1[:, 1], trajectory_1[:, 2])

    if len(trajectory_2):
        ax.plot(trajectory_2[:, 0], trajectory_2[:, 1], trajectory_2[:, 2], '-')

    if save_path:
        plt.savefig(save_path)
        plt.close()


trajectory_function = rossler_system
w_r_function = reservoir_construction_fix_degree
w_i_function = reservoir_construction_fix_degree
w_i_function = reservoir_construction_average_allocate

N = 1200
D = 3
Rou = 1.8
Sigma = 0.1
Alpha = 0.27
Beta = 1e-4

W_r = w_r_function(N, N, 'uniform', D, sr=Rou,  low=0.0, high=Alpha)
W_i = w_i_function(N, 3, 'uniform', low=-Sigma, high=Sigma)

Start_pos = np.array([1, 1, 1])
Training_time = int(5000)
Trajectory_training = trajectory_function(Start_pos, Training_time, 0.01)

Reservoir_start = np.zeros(N)
Reservoir_state_training = np.zeros((len(Trajectory_training), len(Reservoir_start)))
Reservoir_state_training[0, :] = Reservoir_start

for i in tqdm(range(1, len(Trajectory_training))):
    Reservoir_state_training[i, :] = np.tanh(np.dot(W_r, Reservoir_state_training[i - 1, :]) +
                                             np.dot(W_i, Trajectory_training[i - 1, :]))

X = Reservoir_state_training[1000:, :]
Y = Trajectory_training[1000:, :]

S = X.copy()
S[:, 1::2] = S[:, 1::2] ** 2
W_0 = np.linalg.solve(np.dot(S.T, S) + Beta * np.eye(S.shape[1]), np.dot(S.T, Y))
W_0 = W_0.T

W_01 = np.zeros(W_0.shape)
W_02 = np.zeros(W_0.shape)

W_01[:, ::2] = W_0[:, ::2]
W_02[:, 1::2] = W_0[:, 1::2]

Output_training = np.dot(W_01, X.T) + np.dot(W_02, X.T ** 2)
Output_training = Output_training.T


def f_0(r):
    return (np.dot(W_01, r.T) + np.dot(W_02, r.T ** 2)).T


plot_trajectory(Y, Output_training)


Predicting_time = int(5000)
Reservoir_state_predicting = np.zeros((Predicting_time, N))
Trajectory_predicting = np.zeros((Predicting_time, len(Start_pos)))
Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]
Trajectory_predicting[0, :] = Trajectory_training[-1, :]
Trajectory_predicting_original = trajectory_function(Trajectory_training[-1, :], Predicting_time, 0.01)

for i in tqdm(range(1, Predicting_time)):
    Reservoir_state_predicting[i, :] = np.tanh(np.dot(W_r, Reservoir_state_predicting[i - 1, :]) +
                                               np.dot(W_i, Trajectory_predicting[i - 1, :]))
    Trajectory_predicting[i, :] = f_0(Reservoir_state_predicting[i, :])

plot_trajectory(Trajectory_predicting_original, Trajectory_predicting)
