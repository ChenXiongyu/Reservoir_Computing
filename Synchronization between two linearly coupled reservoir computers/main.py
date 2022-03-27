import numpy as np
import reservoir_computing as rc

# Parameters
N = 1200
D = 3
Rou = 1.8
Sigma = 0.1
Alpha = 0.27
Beta = 1e-4

# Mode
Self_predicting = False
Coupled_predicting = True

if Self_predicting:
    # Trajectory for Training
    trajectory_function = rc.rossler_system
    Start_pos = np.array([1, 1, 1])
    Training_time = int(5000)
    Trajectory_training = trajectory_function(Start_pos, Training_time, 0.01)

    # Train Process
    W_r, W_i, F_out, Reservoir_state_training = rc.train(N, D, Rou, Sigma, Alpha, Beta, Trajectory_training,
                                                         plot=True)

    # Trajectory for Predicting
    Predicting_time = int(5000)
    Trajectory_predicting = trajectory_function(Trajectory_training[-1, :], Predicting_time, 0.01)

    # Self Predicting Process
    Reservoir_state_predicting = np.zeros((Predicting_time, N))
    Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]

    Output_predicting = rc.self_predict(W_r, W_i, F_out, Trajectory_predicting, Reservoir_state_predicting,
                                        plot=True)

if Coupled_predicting:
    # Trajectory for Training
    trajectory_function = rc.rossler_system
    Training_time = int(5000)
    Start_pos_training_1 = np.array([1, 1, 1])
    Trajectory_training_1 = trajectory_function(Start_pos_training_1, Training_time, 0.01)
    Start_pos_training_2 = np.array([0.5, 0.5, 0.5])
    Trajectory_training_2 = trajectory_function(Start_pos_training_2, Training_time, 0.01)

    # Train Process
    W_r_1, W_i_1, F_out_1, Reservoir_state_training_1 = rc.train(N, D, Rou, Sigma, Alpha, Beta, Trajectory_training_1,
                                                                 plot=False)
    W_r_2, W_i_2, F_out_2, Reservoir_state_training_2 = rc.train(N, D, Rou, Sigma, Alpha, Beta, Trajectory_training_2,
                                                                 plot=False)
    # Trajectory for Predicting
    Predicting_time = int(5000)
    Start_pos_predicting_1 = Trajectory_training_1[-2, :]
    Trajectory_predicting_1 = trajectory_function(Start_pos_predicting_1, Predicting_time, 0.01)
    Start_pos_predicting_2 = Trajectory_training_2[-2, :]
    Trajectory_predicting_2 = trajectory_function(Start_pos_predicting_2, Predicting_time, 0.01)

    # Coupled Predicting Process
    Coupled_strength = 0.5
    Noise_strength = 0
    Reservoir_state_predicting_1 = np.zeros((Predicting_time, N))
    Reservoir_state_predicting_1[0, :] = Reservoir_state_training_1[-1, :]
    Reservoir_state_predicting_2 = np.zeros((Predicting_time, N))
    Reservoir_state_predicting_2[0, :] = Reservoir_state_training_2[-1, :]

    Output_predicting_1, Output_predicting_2 = \
        rc.coupled_predict(W_r_1, W_i_1, F_out_1, Reservoir_state_predicting_1, Trajectory_predicting_1,
                           W_r_2, W_i_2, F_out_2, Reservoir_state_predicting_2, Trajectory_predicting_2,
                           Coupled_strength, Noise_strength)

    rc.plot_trajectory(Trajectory_predicting_1, Output_predicting_1)
    rc.plot_trajectory(Trajectory_predicting_2, Output_predicting_2)
    rc.plot_trajectory(Output_predicting_1, Output_predicting_2)

    Delta = np.sqrt(np.sum((Output_predicting_1 - Output_predicting_2) ** 2, axis=1))
    print(np.mean(Delta))
