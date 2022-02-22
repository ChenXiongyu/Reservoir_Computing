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


def fix_nonzero_matrix_construction(k, low_limit, up_limit, n_1, n_2, sr=0, scale=0):

    matrix = np.random.uniform(low_limit, up_limit, (n_1, n_2))
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

    r = np.zeros((len(r_0), length + 1))
    r[:, 0] = r_0.T

    for t in range(length):
        derivative = reservoir_derivative(w_r, w_in, r[:, t], trajectory[:, t])
        r[:, t + 1] = r[:, t] + derivative * delta_t

    return r[:, 1:]


def ridge_regression_matrix(reservoir, target, beta):

    coef_matrix = np.dot(reservoir, reservoir.T)
    coef_matrix = coef_matrix + beta * np.eye(len(coef_matrix))
    target_matrix = np.dot(reservoir, target.T)

    matrix = np.linalg.solve(coef_matrix, target_matrix)

    return matrix.T


T_train = 10500
T_sync = 500
N = 200
K = 4
D = 3

W_in = fix_nonzero_matrix_construction(1, -1, 1, N, D)
W_r = fix_nonzero_matrix_construction(K, -1, 1, N, N)

Start_pos = np.array([0, 1, 2])
Delta_t = 0.01
Trajectory = lorenz_63(Start_pos, T_train, Delta_t)

R_0 = np.zeros(N)
Reservoir_state = reservoir_state(W_r, W_in, Trajectory, R_0, Delta_t)

Beta = 0.01
W_out = ridge_regression_matrix(Reservoir_state, Trajectory, Beta)

Prediction = np.dot(W_out, Reservoir_state)

plot_trajectory(Trajectory, Prediction)
