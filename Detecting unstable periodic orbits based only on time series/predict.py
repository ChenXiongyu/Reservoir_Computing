# Prediction
Predicting_length = int(5000)
Trajectory_predicting = trajectory_function(Trajectory[-1, :], Predicting_length, Delta_t)

Output_predicting, Reservoir_predicting_state = output_predicting_function(Reservoir_training_state[-1, :],
                                                                           Trajectory[-1, :], W_r, W_i, F_0,
                                                                           Predicting_length)

plot_trajectory(Trajectory_predicting, Output_predicting)
