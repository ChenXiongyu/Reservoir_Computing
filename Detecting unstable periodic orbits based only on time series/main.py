import numpy as np
import train_predict_valuate

# Train
Trajectory_information = {'start': np.array([1, 1, 1]), 'length': int(5000), 'tick': 0.01}
Parameter = {'n_r': 1200, 'd': 3, 'rou': 0.6, 'alpha': 0.27, 'sigma': 1, 'beta': 1e-4}

Trajectory, Output_training, Reservoir_training_state, F_0, W_r, W_i = \
    train_predict_valuate.train(Trajectory_information, Parameter, plot=True)
RMSE_training = train_predict_valuate.rmse_value(Trajectory, Output_training)

# Predict
Prediction_information = {'start': Trajectory[-1, :], 'length': 5000,
                          'tick': Trajectory_information['tick'],
                          'reservoir_start': Reservoir_training_state[-1, :],
                          'w_r': W_r, 'w_i': W_i, 'f_0': F_0}

Trajectory_predicting, Output_predicting, Reservoir_predicting_state = \
    train_predict_valuate.predict(Prediction_information, plot=True)
RMSE_predicting = train_predict_valuate.rmse_value(Trajectory_predicting, Output_predicting)
