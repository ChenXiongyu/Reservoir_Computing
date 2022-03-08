import numpy as np


def lorenz_system(start_pos, trajectory_length, delta_t):

    trajectory = np.zeros((trajectory_length, len(start_pos)))
    trajectory[0, :] = start_pos

    for t in range(trajectory_length - 1):
        derivative = np.array([10 * (trajectory[t, 1] - trajectory[t, 0]),
                               trajectory[t, 0] * (28 - 10 * trajectory[t, 2]) - trajectory[t, 1],
                               10 * trajectory[t, 0] * trajectory[t, 1] - 8 / 3 * trajectory[t, 2]])
        trajectory[t + 1, :] = trajectory[t, :] + derivative * delta_t

    return trajectory
