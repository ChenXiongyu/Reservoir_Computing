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

    matrix = np.random.randn(n_1, n_2)
    for row in range(n_1):
        index = np.random.choice(n_2, n_2 - k, replace=False)
        matrix[row, :][index] = 0

    if sr:
        sr = sr / max(abs(np.linalg.eigvals(matrix)))
        matrix = sr * matrix

    if scale:
        matrix = scale * matrix

    return matrix


def probability_matrix_construction(sigma, n_1, n_2, scale=0.0):

    matrix = np.random.randn(n_1, n_2) * scale
    index = np.array(np.random.uniform(0, 1, (n_1, n_2)) < sigma, dtype=int)

    return index * matrix


def reservoir_derivative(gamma, w_r, w_in, r, trajectory):
    return -gamma * r + gamma * np.tanh(np.dot(w_r, r) + np.dot(w_in, trajectory))


def reservoir_state(gamma, w_r, w_in, trajectory, r_0, delta_t):

    length = trajectory.shape[1]

    r = np.zeros((len(r_0), length + 1))
    r[:, 0] = r_0.T

    for t in range(length):
        derivative = reservoir_derivative(gamma, w_r, w_in, r[:, t], trajectory[:, t])
        r[:, t + 1] = r[:, t] + derivative * delta_t

    return r[:, 1:]


def reservoir_tilt(reservoir):

    reservoir[int(np.ceil(reservoir.shape[0] / 2)):, :] = reservoir[int(np.ceil(reservoir.shape[0] / 2)):, :] ** 2

    return reservoir


def ridge_regression_matrix(f_out, reservoir, target, beta):

    reservoir = f_out(reservoir)

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
N = 100
D = 3

K = 3
Gamma = 7.7
Sigma = 0.81
Rou_in = 0.37
Rou_r = 0.41


# Initial Matrix Setup
W_in = probability_matrix_construction(Sigma, N, D, scale=Rou_in)
W_r = fix_nonzero_matrix_construction(K, N, N, sr=Rou_r)

# Initial Trajectory Setup
Delta_t = 0.01
Start_pos = np.array([1, 1, 1])

T_discard = int(100 / Delta_t)
T_train = int(100 / Delta_t)

Trajectory = lorenz_63(Start_pos, T_discard + T_train, Delta_t)

# Initial Reservoir State Setup
R_0 = np.zeros(N)
Reservoir_state = reservoir_state(Gamma, W_r, W_in, Trajectory, R_0, Delta_t)

# Training
Trajectory = Trajectory[:, T_discard:]
Reservoir_state = Reservoir_state[:, T_discard:]

Beta = 0.001
W_out = ridge_regression_matrix(reservoir_tilt, Reservoir_state, Trajectory, Beta)
Training = np.dot(W_out, Reservoir_state)
plot_trajectory(Trajectory, Training)

# Prediction
T_test = int(100 / Delta_t)
Prediction_pos = Trajectory[:, -1]
Prediction_trajectory = lorenz_63(Prediction_pos, T_test, Delta_t)

Prediction_r_0 = Reservoir_state[:, -1]
Prediction_reservoir_state = reservoir_state(Gamma, W_r, W_in, Prediction_trajectory, Prediction_r_0, Delta_t)
Prediction_reservoir_state = reservoir_tilt(Prediction_reservoir_state)

Prediction = np.dot(W_out, Prediction_reservoir_state)
plot_trajectory(Prediction_trajectory, Prediction)

print(error_measures(Prediction_trajectory, Prediction))
