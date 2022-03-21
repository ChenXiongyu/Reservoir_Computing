import numpy as np
import matplotlib.pyplot as plt
import trajectory
import reservoir


def plot_trajectory(trajectory_1, trajectory_2=np.array([]), save_path=''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(trajectory_1[:, 0], trajectory_1[:, 1], trajectory_1[:, 2])

    if len(trajectory_2):
        ax.plot(trajectory_2[:, 0], trajectory_2[:, 1], trajectory_2[:, 2], '-')

    if save_path:
        plt.savefig(save_path)
        plt.close()


trajectory_function = trajectory.lorenz_system
w_r_function = reservoir.reservoir_construction_fix_degree
w_i_function = reservoir.reservoir_construction_average_allocate

reservoir_training_function = reservoir.reservoir_training
output_training_function = reservoir.output_training

output_predicting_function = reservoir.output_predicting

Start_pos = np.array([1, 1, 1])
Trajectory_length = int(5000)
Delta_t = 0.01

Trajectory = trajectory_function(Start_pos, Trajectory_length, Delta_t)

N_r = 1200
D = 3
Rou = 0.6
Alpha = 0.27
W_r = w_r_function(N_r, N_r, 'uniform', D, sr=Rou, low=0.0, high=Alpha)

Sigma = 1
W_i = w_i_function(N_r, len(Start_pos), 'uniform', low=-Sigma, high=Sigma)

Reservoir_training_start = np.zeros(N_r)
Reservoir_training_state = reservoir_training_function(W_r, W_i, Reservoir_training_start, Trajectory)

Reservoir_training_state = Reservoir_training_state[1000:, :]
Trajectory = Trajectory[1000:, :]
Beta = 0

s = Reservoir_training_state.copy()
s[:, 1::2] = s[:, 1::2] ** 2
w_0 = np.linalg.solve(np.dot(s.T, s) + Beta * np.eye(s.shape[1]), np.dot(s.T, Trajectory))
w_0 = w_0.T

w_01 = np.zeros(w_0.shape)
w_02 = np.zeros(w_0.shape)

w_01[:, ::2] = w_0[:, ::2]
w_02[:, 1::2] = w_0[:, 1::2]

output = np.dot(w_01, Reservoir_training_state.T) + np.dot(w_02, Reservoir_training_state.T ** 2)

plot_trajectory(Trajectory, output)

# Output_training, F_0 = output_training_function(Reservoir_training_state, Trajectory, Beta)
#
# plot_trajectory(Trajectory, Output_training)
