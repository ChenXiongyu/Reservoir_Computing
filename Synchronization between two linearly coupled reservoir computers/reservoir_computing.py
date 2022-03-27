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


def train(n, d, rou, sigma, alpha, beta, trajectory_training, plot=True):
    print('Train Process...')
    w_r_function = reservoir_construction_fix_degree
    w_i_function = reservoir_construction_average_allocate

    w_r = w_r_function(n, n, 'uniform', d, sr=rou,  low=0.0, high=alpha)
    w_i = w_i_function(n, 3, 'uniform', low=-sigma, high=sigma)

    reservoir_start = np.zeros(n)
    reservoir_state_training = np.zeros((len(trajectory_training), len(reservoir_start)))
    reservoir_state_training[0, :] = reservoir_start

    for i in tqdm(range(1, len(trajectory_training))):
        reservoir_state_training[i, :] = np.tanh(np.dot(w_r, reservoir_state_training[i - 1, :]) +
                                                 np.dot(w_i, trajectory_training[i - 1, :]))

    x = reservoir_state_training[1000:, :]
    y = trajectory_training[1000:, :]

    s = x.copy()
    s[:, 1::2] = s[:, 1::2] ** 2
    w_0 = np.linalg.solve(np.dot(s.T, s) + beta * np.eye(s.shape[1]), np.dot(s.T, y))
    w_0 = w_0.T

    w_01 = np.zeros(w_0.shape)
    w_02 = np.zeros(w_0.shape)

    w_01[:, ::2] = w_0[:, ::2]
    w_02[:, 1::2] = w_0[:, 1::2]

    output_training = np.dot(w_01, x.T) + np.dot(w_02, x.T ** 2)
    output_training = output_training.T

    def f_out(r):
        return (np.dot(w_01, r.T) + np.dot(w_02, r.T ** 2)).T

    if plot:
        plot_trajectory(y, output_training)

    return w_r, w_i, f_out, reservoir_state_training


def self_predict(w_r, w_i, f_out, trajectory_predicting, reservoir_state_predicting, plot=True):
    print('Self Predicting Process...')
    output_predicting = np.zeros(trajectory_predicting.shape)
    output_predicting[0, :] = trajectory_predicting[0, :]

    for i in tqdm(range(1, len(trajectory_predicting))):
        reservoir_state_predicting[i, :] = np.tanh(np.dot(w_r, reservoir_state_predicting[i - 1, :]) +
                                                   np.dot(w_i, output_predicting[i - 1, :]))
        output_predicting[i, :] = f_out(reservoir_state_predicting[i, :])

    if plot:
        plot_trajectory(trajectory_predicting, output_predicting)

    return output_predicting


def coupled_predict(w_r_1, w_i_1, f_out_1, reservoir_state_predicting_1, trajectory_predicting_1,
                    w_r_2, w_i_2, f_out_2, reservoir_state_predicting_2, trajectory_predicting_2,
                    coupled_strength, noise_strength):
    print('Coupled Predicting Process...')
    output_predicting_1 = np.zeros(trajectory_predicting_1.shape)
    output_predicting_2 = np.zeros(trajectory_predicting_2.shape)
    output_predicting_1[0, :] = trajectory_predicting_1[0, :]
    output_predicting_2[0, :] = trajectory_predicting_2[0, :]

    for i in tqdm(range(1, len(trajectory_predicting_1))):
        reservoir_state_predicting_1[i, :] = \
            np.tanh(np.dot(w_r_1, reservoir_state_predicting_1[i - 1, :]) +
                    np.dot(w_i_1,
                           coupled_strength * output_predicting_2[i - 1, :] +
                           (1 - coupled_strength) * output_predicting_1[i - 1, :] +
                           noise_strength * np.random.rand(3)))
        reservoir_state_predicting_2[i, :] = \
            np.tanh(np.dot(w_r_2, reservoir_state_predicting_2[i - 1, :]) +
                    np.dot(w_i_2,
                           coupled_strength * output_predicting_1[i - 1, :] +
                           (1 - coupled_strength) * output_predicting_2[i - 1, :] +
                           noise_strength * np.random.rand(3)))

        output_predicting_1[i, :] = f_out_1(reservoir_state_predicting_1[i, :])
        output_predicting_2[i, :] = f_out_2(reservoir_state_predicting_2[i, :])

    return output_predicting_1, output_predicting_2


def train_teacher(n, d, rou, sigma, alpha, beta, trajectory_training, plot=True):
    print('Train (Teacher) Process...')
    w_r_function = reservoir_construction_fix_degree
    w_i_function = reservoir_construction_average_allocate

    w_r = w_r_function(n, n, 'uniform', d, sr=rou,  low=0.0, high=alpha)
    w_i = w_i_function(n, 3, 'uniform', low=-sigma, high=sigma)

    reservoir_start = np.zeros(n)
    reservoir_state_training = np.zeros((len(trajectory_training), len(reservoir_start)))
    reservoir_state_training[0, :] = reservoir_start

    for i in tqdm(range(1, len(trajectory_training))):
        reservoir_state_training[i, :] = np.tanh(np.dot(w_r, reservoir_state_training[i - 1, :]) +
                                                 np.dot(w_i, trajectory_training[i - 1, :]))

    x = np.hstack((trajectory_training[:-1, :], reservoir_state_training[1:, :]))[999:, :]
    y = trajectory_training[1000:, :]

    s = x.copy()
    s[:, 1::2] = s[:, 1::2] ** 2
    w_0 = np.linalg.solve(np.dot(s.T, s) + beta * np.eye(s.shape[1]), np.dot(s.T, y))
    w_0 = w_0.T

    w_01 = np.zeros(w_0.shape)
    w_02 = np.zeros(w_0.shape)

    w_01[:, ::2] = w_0[:, ::2]
    w_02[:, 1::2] = w_0[:, 1::2]

    output_training = np.dot(w_01, x.T) + np.dot(w_02, x.T ** 2)
    output_training = output_training.T

    def f_out(r):
        return (np.dot(w_01, r.T) + np.dot(w_02, r.T ** 2)).T

    if plot:
        plot_trajectory(y, output_training)

    return w_r, w_i, f_out, reservoir_state_training


def self_predict_teacher(w_r, w_i, f_out, trajectory_predicting, reservoir_state_predicting, plot=True):
    print('Self Predicting (Teacher) Process...')
    output_predicting = np.zeros(trajectory_predicting.shape)
    output_predicting[0, :] = trajectory_predicting[0, :]

    for i in tqdm(range(1, len(trajectory_predicting))):
        reservoir_state_predicting[i, :] = np.tanh(np.dot(w_r, reservoir_state_predicting[i - 1, :]) +
                                                   np.dot(w_i, output_predicting[i - 1, :]))
        output_predicting[i, :] = f_out(np.append(output_predicting[i - 1, :], reservoir_state_predicting[i, :]))

    if plot:
        plot_trajectory(trajectory_predicting, output_predicting)

    return output_predicting