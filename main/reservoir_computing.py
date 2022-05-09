import numpy as np
import matplotlib.pyplot as plt
from nolitsa import lyapunov, data
from tqdm import tqdm

from outer_function import poly_fit


# Data Module
lorenz = data.lorenz  # Lorenz System

roessler = data.roessler  # Roessler System


def sprott(length, sample, x0, discard=0):
    
    time = np.linspace(sample, sample * length, length)
    
    def sprott_ode(y, _):
        x, y, z = y
        dydt = [y * z, x - y, 1 - x * y]
        return dydt
    
    from scipy.integrate import odeint
    sol = odeint(sprott_ode, x0, time)  
    sol = sol[discard:, :]
    
    return time, sol
    
    
# Reservoir Construction Module
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

    elif scale:
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


def reservoir_construction_probability_symmetry(n_1, n_2, random_type, probability, symmetry, antisymmetry,
                                                low=0.0, high=0.0, scale=0.0, sr=0.0):
    if symmetry + antisymmetry > 1:
        return np.zeros((n_1, n_2))

    symmetry = probability * symmetry
    antisymmetry = probability * antisymmetry
    non_symmetry = probability - symmetry - antisymmetry

    while True:
        matrix_symmetry = initial_reservoir(n_1, n_2, random_type, low=low, high=high)
        index_symmetry = np.array(np.random.uniform(0, 1, (n_1, n_2)) < symmetry, dtype=int)
        matrix_symmetry = index_symmetry * matrix_symmetry
        matrix_symmetry = np.triu(matrix_symmetry, 1).T + np.triu(matrix_symmetry)

        matrix_antisymmetry = initial_reservoir(n_1, n_2, random_type, low=low, high=high)
        index_antisymmetry = np.array(np.random.uniform(0, 1, (n_1, n_2)) < antisymmetry, dtype=int)
        matrix_antisymmetry = index_antisymmetry * matrix_antisymmetry
        matrix_antisymmetry = np.triu(matrix_antisymmetry, 1).T - np.triu(matrix_symmetry, 1)

        matrix_non_symmetry = initial_reservoir(n_1, n_2, random_type, low=low, high=high)
        index_non_symmetry = np.array(np.random.uniform(0, 1, (n_1, n_2)) < non_symmetry, dtype=int)
        matrix_non_symmetry = index_non_symmetry * matrix_non_symmetry

        reservoir = matrix_symmetry + matrix_antisymmetry + matrix_non_symmetry

        if sr:
            eig = max(abs(np.linalg.eigvals(reservoir)))
            if eig < 1e-4:
                continue
            else:
                sr = sr / eig
            reservoir = sr * reservoir
            break

    if scale:
        reservoir = scale * reservoir

    return reservoir


# Activation Function Module
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


# Basis Function Module
def original(array):
    return array


def square(array):
    return np.square(array)


def sin(array, k=1):
    return np.sin(k * array)


def cos(array, k=1):
    return np.cos(k * array)


# Plot Module
def plot_trajectory(trajectory_1, trajectory_2=np.array([]), save_path=''):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(trajectory_1[:, 0], trajectory_1[:, 1], trajectory_1[:, 2])

    if len(trajectory_2):
        ax.plot(trajectory_2[:, 0], trajectory_2[:, 1], trajectory_2[:, 2], '-')

    if save_path:
        plt.savefig(save_path, format='svg')
        plt.close()


# Computing Module
def train(n, d, rou, sigma, beta, trajectory_training, plot=True,
          activation_function=np.tanh, 
          basis_function_1=original, basis_function_2=np.square, 
          discard=1000):
    # print('Train Process...')
    w_r_function = reservoir_construction_fix_degree
    w_i_function = reservoir_construction_average_allocate

    w_r = w_r_function(n, n, 'uniform', d, sr=rou,  low=0.0, high=1.0)
    w_i = w_i_function(n, d, 'uniform', low=-sigma, high=sigma)

    reservoir_start = np.zeros(n)
    reservoir_state_training = np.zeros((len(trajectory_training), len(reservoir_start)))
    reservoir_state_training[0, :] = reservoir_start

    for i in range(1, len(trajectory_training)):
        reservoir_state_training[i, :] = activation_function(np.dot(w_r, reservoir_state_training[i - 1, :]) +
                                                             np.dot(w_i, trajectory_training[i - 1, :]))

    x = reservoir_state_training[discard:, :]
    y = trajectory_training[discard:, :]

    s = x.copy()
    s[:, ::2] = basis_function_1(s[:, ::2])
    s[:, 1::2] = basis_function_2(s[:, 1::2])
    w_0 = np.linalg.solve(np.dot(s.T, s) + beta * np.eye(s.shape[1]), np.dot(s.T, y))
    w_0 = w_0.T

    w_01 = np.zeros(w_0.shape)
    w_02 = np.zeros(w_0.shape)

    w_01[:, ::2] = w_0[:, ::2]
    w_02[:, 1::2] = w_0[:, 1::2]

    output_training = np.dot(w_01, basis_function_1(x.T)) + np.dot(w_02, basis_function_2(x.T))
    output_training = output_training.T

    def f_out(r):
        return (np.dot(w_01, basis_function_1(r.T)) + np.dot(w_02, basis_function_2(r.T))).T

    if plot:
        plot_trajectory(y, output_training)

    return w_r, w_i, f_out, reservoir_state_training, output_training


def self_predict(w_r, w_i, f_out, trajectory_predicting, reservoir_state_predicting,
                 activation_function=np.tanh, plot=True, save_path=''):
    # print('Self Predicting Process...')
    output_predicting = np.zeros(trajectory_predicting.shape)
    output_predicting[0, :] = trajectory_predicting[0, :]

    for i in range(1, len(trajectory_predicting)):
        reservoir_state_predicting[i, :] = activation_function(np.dot(w_r, reservoir_state_predicting[i - 1, :]) +
                                                               np.dot(w_i, output_predicting[i - 1, :]))
        output_predicting[i, :] = f_out(reservoir_state_predicting[i, :])

    if plot:
        plot_trajectory(trajectory_predicting, output_predicting)
        
        if save_path:
            plt.savefig(save_path, format='svg')
            plt.close()

    return output_predicting


# Evaluation Module
def error_evaluate(trajectory_target, trajectory_output, time, time_start=0, time_end=0, plot=True, save_path=''):
    difference = trajectory_target - trajectory_output
    if time_end == 0:
        time_end = min(len(trajectory_target), len(trajectory_output))

    distance = np.sqrt(np.sum(difference[time_start:time_end, :] ** 2, axis=1))

    rmse = np.sqrt(np.mean(np.sum(difference[time_start:time_end, :] ** 2, axis=1)))
    nrmse = np.sqrt(np.sum(difference[time_start:time_end, :] ** 2) /
                    np.sum((trajectory_target - np.mean(trajectory_target, axis=0))[time_start:time_end, :] ** 2))
    mape = float(np.mean(distance / np.sqrt(np.sum(trajectory_target[time_start:time_end, :] ** 2, axis=1))))

    if plot:
        plt.figure()
        plt.plot(time[time_start:time_end], distance)
        plt.text(0, max(distance) / 2, 'RMSE = %.2f\nNRMSE = %.2f\nMAPE = %.2f' % (rmse, nrmse, mape))

        if save_path:
            plt.savefig(save_path, format='svg')
            plt.close()
        
    result = {'RMSE': rmse, 'nrmse': nrmse, 'mape': mape}
    return distance, result





def coupled_predict(w_r_1, w_i_1, f_out_1, reservoir_state_predicting_1, trajectory_predicting_1,
                    w_r_2, w_i_2, f_out_2, reservoir_state_predicting_2, trajectory_predicting_2,
                    coupled_strength, noise_strength, activation_function=np.tanh):
    output_predicting_1 = np.zeros(trajectory_predicting_1.shape)
    output_predicting_2 = np.zeros(trajectory_predicting_2.shape)
    output_predicting_1[0, :] = trajectory_predicting_1[0, :]
    output_predicting_2[0, :] = trajectory_predicting_2[0, :]

    for i in range(1, len(trajectory_predicting_1)):
        reservoir_state_predicting_1[i, :] = \
            activation_function(np.dot(w_r_1, reservoir_state_predicting_1[i - 1, :]) +
                                np.dot(w_i_1,
                                       coupled_strength * output_predicting_2[i - 1, :] +
                                       (1 - coupled_strength) * output_predicting_1[i - 1, :] +
                                       noise_strength * np.random.rand(3)))
        reservoir_state_predicting_2[i, :] = \
            activation_function(np.dot(w_r_2, reservoir_state_predicting_2[i - 1, :]) +
                                np.dot(w_i_2,
                                       coupled_strength * output_predicting_1[i - 1, :] +
                                       (1 - coupled_strength) * output_predicting_2[i - 1, :] +
                                       noise_strength * np.random.rand(3)))

        output_predicting_1[i, :] = f_out_1(reservoir_state_predicting_1[i, :])
        output_predicting_2[i, :] = f_out_2(reservoir_state_predicting_2[i, :])

    return output_predicting_1, output_predicting_2


def train_teacher(n, d, rou, sigma, alpha, beta, trajectory_training, activation_function=np.tanh, plot=True):
    print('Train (Teacher) Process...')
    w_r_function = reservoir_construction_fix_degree
    w_i_function = reservoir_construction_average_allocate

    w_r = w_r_function(n, n, 'uniform', d, sr=rou,  low=0.0, high=alpha)
    w_i = w_i_function(n, 3, 'uniform', low=-sigma, high=sigma)

    reservoir_start = np.zeros(n)
    reservoir_state_training = np.zeros((len(trajectory_training), len(reservoir_start)))
    reservoir_state_training[0, :] = reservoir_start

    for i in tqdm(range(1, len(trajectory_training))):
        reservoir_state_training[i, :] = activation_function(np.dot(w_r, reservoir_state_training[i - 1, :]) +
                                                             np.dot(w_i, trajectory_training[i - 1, :]))

    x = np.hstack((trajectory_training[:-1, :], reservoir_state_training[1:, :]))[999:, :]
    y = trajectory_training[1000:, :]

    s = x.copy()
    w_0 = np.linalg.solve(np.dot(s.T, s) + beta * np.eye(s.shape[1]), np.dot(s.T, y))
    w_0 = w_0.T

    output_training = np.dot(w_0, x.T)
    output_training = output_training.T

    def f_out(r):
        return (np.dot(w_0, r.T)).T

    if plot:
        plot_trajectory(y, output_training)

    return w_r, w_i, f_out, reservoir_state_training


def self_predict_teacher(w_r, w_i, f_out, trajectory_predicting, reservoir_state_predicting,
                         activation_function=np.tanh, plot=True):
    print('Self Predicting (Teacher) Process...')
    output_predicting = np.zeros(trajectory_predicting.shape)
    output_predicting[0, :] = trajectory_predicting[0, :]

    for i in tqdm(range(1, len(trajectory_predicting))):
        reservoir_state_predicting[i, :] = activation_function(np.dot(w_r, reservoir_state_predicting[i - 1, :]) +
                                                               np.dot(w_i, output_predicting[i - 1, :]))
        output_predicting[i, :] = f_out(np.append(output_predicting[i - 1, :], reservoir_state_predicting[i, :]))

    if plot:
        plot_trajectory(trajectory_predicting, output_predicting)

    return output_predicting


def lle_lorenz(trajectory, dt=0.01, maxt=250, window=30):
    divergence = lyapunov.mle(trajectory, maxt=maxt, window=window)
    max_t = np.arange(maxt) * dt
    coef = poly_fit(max_t, divergence, 1)[0]
    return coef


def train_reservoir(n, rou, sigma, alpha, beta, probability, symmetry, antisymmetry, trajectory_training, plot=True,
                    activation_function=np.tanh, basis_function_1=original, basis_function_2=np.square):
    # print('Train Process...')
    w_r_function = reservoir_construction_probability_symmetry
    w_i_function = reservoir_construction_average_allocate

    w_r = w_r_function(n, n, 'uniform', probability, symmetry, antisymmetry, low=0.0, high=alpha, sr=rou)
    w_i = w_i_function(n, 3, 'uniform', low=-sigma, high=sigma)

    reservoir_start = np.zeros(n)
    reservoir_state_training = np.zeros((len(trajectory_training), len(reservoir_start)))
    reservoir_state_training[0, :] = reservoir_start

    for i in range(1, len(trajectory_training)):
        reservoir_state_training[i, :] = activation_function(np.dot(w_r, reservoir_state_training[i - 1, :]) +
                                                             np.dot(w_i, trajectory_training[i - 1, :]))

    x = reservoir_state_training[1000:, :]
    y = trajectory_training[1000:, :]

    s = x.copy()
    s[:, ::2] = basis_function_1(s[:, ::2])
    s[:, 1::2] = basis_function_2(s[:, 1::2])
    w_0 = np.linalg.solve(np.dot(s.T, s) + beta * np.eye(s.shape[1]), np.dot(s.T, y))
    w_0 = w_0.T

    w_01 = np.zeros(w_0.shape)
    w_02 = np.zeros(w_0.shape)

    w_01[:, ::2] = w_0[:, ::2]
    w_02[:, 1::2] = w_0[:, 1::2]

    output_training = np.dot(w_01, basis_function_1(x.T)) + np.dot(w_02, basis_function_2(x.T))
    output_training = output_training.T

    def f_out(r):
        return (np.dot(w_01, basis_function_1(r.T)) + np.dot(w_02, basis_function_2(r.T))).T

    if plot:
        plot_trajectory(y, output_training)

    return w_r, w_i, f_out, reservoir_state_training, output_training


def train_delay(n, d, rou, sigma, alpha, beta, delay, trajectory_training, plot=True,
                activation_function=np.tanh, basis_function_1=original, basis_function_2=np.square):
    # print('Train Process...')
    w_r_function = reservoir_construction_fix_degree
    w_i_function = reservoir_construction_average_allocate

    w_r = w_r_function(n, n, 'uniform', d, sr=rou,  low=0.0, high=alpha)
    w_i = w_i_function(n, 3, 'uniform', low=-sigma, high=sigma)

    reservoir_start = np.zeros(n)
    reservoir_state_training = np.zeros((len(trajectory_training), len(reservoir_start)))
    reservoir_state_training[0, :] = reservoir_start

    for i in range(1, len(trajectory_training) - delay):
        reservoir_state_training[i + delay, :] = activation_function(
            np.dot(w_r, reservoir_state_training[i - 1, :]) + np.dot(w_i, trajectory_training[i - 1, :]))

    x = reservoir_state_training[1000:, :]
    y = trajectory_training[1000:, :]

    s = x.copy()
    s[:, ::2] = basis_function_1(s[:, ::2])
    s[:, 1::2] = basis_function_2(s[:, 1::2])
    w_0 = np.linalg.solve(np.dot(s.T, s) + beta * np.eye(s.shape[1]), np.dot(s.T, y))
    w_0 = w_0.T

    w_01 = np.zeros(w_0.shape)
    w_02 = np.zeros(w_0.shape)

    w_01[:, ::2] = w_0[:, ::2]
    w_02[:, 1::2] = w_0[:, 1::2]

    output_training = np.dot(w_01, basis_function_1(x.T)) + np.dot(w_02, basis_function_2(x.T))
    output_training = output_training.T

    def f_out(r):
        return (np.dot(w_01, basis_function_1(r.T)) + np.dot(w_02, basis_function_2(r.T))).T

    if plot:
        plot_trajectory(y, output_training)

    return w_r, w_i, f_out, reservoir_state_training, output_training


def self_predict_delay(w_r, w_i, f_out, delay, trajectory_predicting, reservoir_state_predicting,
                       activation_function=np.tanh, plot=True):
    # print('Self Predicting Process...')
    output_predicting = np.zeros(trajectory_predicting.shape)
    output_predicting[0:(delay + 1), :] = trajectory_predicting[0:(delay + 1), :]

    for i in range(1, len(trajectory_predicting) - delay):
        reservoir_state_predicting[i + delay, :] = activation_function(
            np.dot(w_r, reservoir_state_predicting[i - 1, :]) + np.dot(w_i, output_predicting[i - 1, :]))
        output_predicting[i + delay, :] = f_out(reservoir_state_predicting[i + delay, :])

    if plot:
        plot_trajectory(trajectory_predicting, output_predicting)

    return output_predicting
