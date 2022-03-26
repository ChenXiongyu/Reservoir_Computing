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


trajectory_function = trajectory.rossler_system
w_r_function = reservoir.reservoir_construction_fix_degree
w_i_function = reservoir.reservoir_construction_average_allocate

reservoir_training_function = reservoir.reservoir_training
output_training_function = reservoir.output_training

output_predicting_function = reservoir.output_predicting


def train(trajectory_information, parameter_information, plot=True):
    print('Train Process')
    start_pos = trajectory_information['start']
    trajectory_length = trajectory_information['length']
    delta_t = trajectory_information['tick']
    traj = trajectory_function(start_pos, trajectory_length, delta_t)

    n_r = parameter_information['n_r']
    d = parameter_information['d']
    rou = parameter_information['rou']
    alpha = parameter_information['alpha']

    w_r = w_r_function(n_r, n_r, 'uniform', d, sr=rou, low=0.0, high=alpha)

    sigma = parameter_information['sigma']
    w_i = w_i_function(n_r, len(start_pos), 'uniform', low=-sigma, high=sigma)

    reservoir_training_start = np.zeros(n_r)
    reservoir_training_state = reservoir_training_function(w_r, w_i, reservoir_training_start, traj)

    reservoir_training_state = reservoir_training_state[1000:, :]
    traj = traj[1000:, :]
    beta = parameter_information['beta']

    output_training, f_0 = output_training_function(reservoir_training_state, traj, beta)

    if plot:
        plot_trajectory(traj, output_training)

    return traj, output_training, reservoir_training_state, f_0, w_r, w_i


def predict(prediction_information, plot=True):
    print('Predict Process')
    predicting_length = prediction_information['length']
    delta_t = prediction_information['tick']
    trajectory_predicting = trajectory_function(prediction_information['start'], predicting_length, delta_t)

    output_predicting, reservoir_predicting_state = \
        output_predicting_function(prediction_information['reservoir_start'], prediction_information['start'],
                                   prediction_information['w_r'], prediction_information['w_i'],
                                   prediction_information['f_0'], predicting_length)

    if plot:
        plot_trajectory(trajectory_predicting, output_predicting)

    return trajectory_predicting, output_predicting, reservoir_predicting_state


def rmse_value(trajectory_1, trajectory_2):
    distance = np.sqrt(np.sum((trajectory_1 - trajectory_2) ** 2, axis=1))
    rmse = np.mean(distance)
    plt.figure()
    plt.plot(distance)
    plt.title('RMSE = %f' % rmse)

    return rmse
