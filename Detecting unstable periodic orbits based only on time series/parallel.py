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


def rmse_value(trajectory_1, trajectory_2):
    distance = np.sqrt(np.sum((trajectory_1 - trajectory_2) ** 2, axis=1))
    rmse = np.mean(distance)
    plt.figure()
    plt.plot(distance)
    plt.title('RMSE = %f' % rmse)

    return rmse


trajectory_function = trajectory.lorenz_system
w_r_function = reservoir.reservoir_construction_fix_degree
w_i_function = reservoir.reservoir_construction_average_allocate

reservoir_training_function = reservoir.reservoir_training
output_training_function = reservoir.output_training

output_predicting_function = reservoir.output_predicting

Trajectory_num = 1
Trajectory = {}
Trajectory_length = int(5000)
Delta_t = 0.01
for i in range(Trajectory_num):
    Start_pos = (i + 1) * 0.1 * np.array([1, 1, 1])

    Trajectory[i] = trajectory_function(Start_pos, Trajectory_length, Delta_t)
Trajectory = np.hstack([Trajectory[i] for i in range(Trajectory_num)])
Start_pos = Trajectory[0, :]

N_r = 1200
D = 3
Rou = 0.6
Alpha = 0.27
W_r = w_r_function(N_r, N_r, 'uniform', D, sr=Rou, low=0.0, high=Alpha)

Sigma = 1
W_i = w_i_function(N_r, len(Start_pos), 'uniform', low=-Sigma, high=Sigma)

print('Train Process')
Reservoir_training_start = np.zeros(N_r)
Reservoir_training_state = reservoir_training_function(W_r, W_i, Reservoir_training_start, Trajectory)

Reservoir_training_state = Reservoir_training_state[1000:, :]
Trajectory = Trajectory[1000:, :3]
Beta = 1e-4

Output_training, F_0 = output_training_function(Reservoir_training_state, Trajectory, Beta)

plot_trajectory(Trajectory, Output_training)

# Prediction
print('Predict Process')
Predicting_length = int(1000)
Trajectory_predicting = trajectory_function(Trajectory[-1, :3], Predicting_length, Delta_t)

Output_predicting, Reservoir_predicting_state = output_predicting_function(Reservoir_training_state[-1, :],
                                                                           Trajectory[-1, :3], W_r, W_i[:, :3], F_0,
                                                                           Predicting_length)

plot_trajectory(Trajectory_predicting, Output_predicting)
rmse_value(Trajectory_predicting, Output_predicting)
