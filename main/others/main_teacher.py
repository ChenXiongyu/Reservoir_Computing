import numpy as np
import reservoir_computing as rc

# Parameters
N = 1200
D = 3
Rou = 1.8
Sigma = 0.1
Alpha = 0.27
Beta = 1e-4

# Trajectory for Training
trajectory_function = rc.rossler_system
Start_pos = np.array([1, 1, 1])
Training_time = int(5000)
Trajectory_training = trajectory_function(Start_pos, Training_time, 0.01)

# Train Process
W_r, W_i, F_out, Reservoir_state_training = rc.train_teacher(N, D, Rou, Sigma, Alpha, Beta,
                                                             Trajectory_training, plot=True)

# Trajectory for Predicting
Predicting_time = int(5000)
Trajectory_predicting = trajectory_function(Trajectory_training[-1, :], Predicting_time, 0.01)

# Self Predicting Process
Reservoir_state_predicting = np.zeros((Predicting_time, N))
Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]

Output_predicting = rc.self_predict_teacher(W_r, W_i, F_out, Trajectory_predicting,
                                            Reservoir_state_predicting, plot=True)
