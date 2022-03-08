# Reservoir Computing and its Sensitivity to Symmetry in the Activation Function
# 15 October 2020

import numpy as np
import matplotlib.pyplot as plt


def lorenz_63(start_pos, time_tick, delta_t):

    trajectory = np.zeros((len(start_pos), time_tick + 1))
    trajectory[:, 0] = start_pos

    for t in range(time_tick):
        derivative = np.array([10 * (trajectory[1, t] - trajectory[0, t]),
                               trajectory[0, t] * (28 - trajectory[2, t]) - trajectory[1, t],
                               trajectory[0, t] * trajectory[1, t] - 8 / 3 * trajectory[2, t]])
        trajectory[:, t + 1] = trajectory[:, t] + derivative * delta_t

    return trajectory


def plot_trajectory(trajectory_1, trajectory_2=np.array([]), save_path=''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(trajectory_1[0, :], trajectory_1[1, :], trajectory_1[2, :])

    if len(trajectory_2):
        ax.plot(trajectory_2[0, :], trajectory_2[1, :], trajectory_2[2, :], '-')

    if save_path:
        plt.savefig(save_path)
        plt.close()


def fix_nonzero_matrix_construction(k, n_1, n_2, sr=0.0, scale=0.0):

    matrix = np.random.uniform(-1, 1, (n_1, n_2))
    for row in range(n_1):
        index = np.random.choice(n_2, n_2 - k, replace=False)
        matrix[row, :][index] = 0

    if sr:
        sr = sr / max(abs(np.linalg.eigvals(matrix)))
        matrix = sr * matrix

    if scale:
        matrix = scale * matrix

    return matrix


def reservoir_derivative(w_r, w_in, r, trajectory):
    return np.tanh(np.dot(w_r, r) + np.dot(w_in, trajectory))


def reservoir_state(w_r, w_in, trajectory, r_0, delta_t):

    length = trajectory.shape[1]

    r = np.zeros((len(r_0), length))
    r[:, 0] = r_0.T

    for t in range(length - 1):
        derivative = reservoir_derivative(w_r, w_in, r[:, t], trajectory[:, t])
        r[:, t + 1] = r[:, t] + derivative * delta_t

    return r


def ridge_regression_matrix(reservoir, target, beta):

    coef_matrix = np.dot(reservoir, reservoir.T)
    coef_matrix = coef_matrix + beta * np.eye(len(coef_matrix))
    target_matrix = np.dot(reservoir, target.T)

    matrix = np.linalg.solve(coef_matrix, target_matrix)

    return matrix.T


def error_measures(trajectory_1, trajectory_2):

    length = trajectory_2.shape[1]

    index = np.array(np.linspace(0, length - 1, 50), dtype=int)
    epsilon = np.sqrt(np.sum((trajectory_1[:, index] - trajectory_2[:, index]) ** 2) / 50)

    return epsilon


# Initial Setup
N = 1000
D = 3

A = 0.32
Epsilon = 0.5
S_input = A * (1 - Epsilon)
Rou = A * Epsilon

# Initial Matrix Setup
W_in = fix_nonzero_matrix_construction(1, N, D, scale=S_input)
W_r = fix_nonzero_matrix_construction(4, N, N, sr=Rou)

# Initial Trajectory Setup
Delta_t = 0.02
Start_pos = np.array([0.1, 0.1, 0.1])

T_discard = 500
T_train = 10000
Trajectory = lorenz_63(Start_pos, T_discard + T_train, Delta_t)

# Initial Reservoir State Setup
R_0 = np.zeros(N)
Reservoir_state = reservoir_state(W_r, W_in, Trajectory, R_0, Delta_t)

# Training
Trajectory = Trajectory[:, T_discard:]
Reservoir_state = Reservoir_state[:, T_discard:]

Beta = 1e-9
W_out = ridge_regression_matrix(Reservoir_state, Trajectory, Beta)
Training = np.dot(W_out, Reservoir_state)
plot_trajectory(Trajectory, Training)
print(error_measures(Trajectory, Training))

# Prediction
T_test = 200

Prediction_pos = Trajectory[:, -1]
Prediction_trajectory = lorenz_63(Prediction_pos, T_test, Delta_t)

Prediction_r_0 = Reservoir_state[:, -1]
Prediction_reservoir_state = np.zeros((len(Prediction_r_0), T_test + 1))
Prediction_reservoir_state[:, 0] = Prediction_r_0

for Tick in range(T_test):

    Prediction = np.dot(W_out, Prediction_reservoir_state[:, Tick])

    Derivative = reservoir_derivative(W_r, W_in, Prediction_reservoir_state[:, Tick], Prediction)
    Prediction_reservoir_state[:, Tick + 1] = Prediction_reservoir_state[:, Tick] + Derivative * Delta_t

Prediction = np.dot(W_out, Prediction_reservoir_state)

plot_trajectory(Prediction_trajectory, Prediction)

print(error_measures(Prediction_trajectory, Prediction))
