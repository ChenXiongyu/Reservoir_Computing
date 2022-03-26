import numpy as np
import train_predict_valuate

# Train
Trajectory_information = {'start': np.array([1, 1, 1]), 'length': int(5000), 'tick': 0.01}
Parameter = {'n_r': 1200, 'd': 3, 'rou': 0.6, 'alpha': 0.27, 'sigma': 1, 'beta': 1e-4}

Trajectory_1, Output_training_1, Reservoir_training_state_1, F_0_1, W_r_1, W_i_1 = \
    train_predict_valuate.train(Trajectory_information, Parameter, plot=True)
RMSE_training = train_predict_valuate.rmse_value(Trajectory_1, Output_training_1)

Trajectory_information = {'start': np.array([0.5, 0.5, 0.5]), 'length': int(5000), 'tick': 0.01}
Parameter = {'n_r': 1200, 'd': 3, 'rou': 0.6, 'alpha': 0.27, 'sigma': 1, 'beta': 1e-4}

Trajectory_2, Output_training_2, Reservoir_training_state_2, F_0_2, W_r_2, W_i_2 = \
    train_predict_valuate.train(Trajectory_information, Parameter, plot=True)
RMSE_training = train_predict_valuate.rmse_value(Trajectory_2, Output_training_2)

# Predict
Prediction_information = {'start': np.array([0.75, 0.75, 0.75]), 'length': 5000,
                          'tick': Trajectory_information['tick'],
                          'reservoir_start': np.zeros((1, 1200)),
                          'w_r': W_r, 'w_i': W_i, 'f_0': F_0}

print('Predict Process')
predicting_length = Prediction_information['length']
delta_t = Prediction_information['tick']
trajectory_predicting = trajectory_function(prediction_information['start'], predicting_length, delta_t)

output_predicting, reservoir_predicting_state = \
    output_predicting_function(prediction_information['reservoir_start'], prediction_information['start'],
                               prediction_information['w_r'], prediction_information['w_i'],
                               prediction_information['f_0'], predicting_length)

if plot:
    plot_trajectory(trajectory_predicting, output_predicting)

Trajectory_predicting, Output_predicting, Reservoir_predicting_state = \
    train_predict_valuate.predict(Prediction_information, plot=True)
RMSE_predicting = train_predict_valuate.rmse_value(Trajectory_predicting, Output_predicting)
