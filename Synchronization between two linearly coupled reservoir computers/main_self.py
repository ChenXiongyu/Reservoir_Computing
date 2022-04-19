import warnings

import numpy as np
from nolitsa import data

import activation_function as ac
import basis_function as bf
import reservoir_computing as rc

warnings.filterwarnings('ignore')

# Parameters
N = 1000
D = 3
Alpha = 1  # Useless Actually
Beta = 1e-4
Sigma = 1
Rou = 0.9

# Activation Function
Activation_function = ac.relu

# Trajectory for Training
Start_pos = list(np.random.rand(3))
Time_training, Trajectory_training = data.lorenz(length=5555, sample=0.01, x0=Start_pos, discard=0,
                                                 sigma=10, beta=8/3, rho=28)

# Train Process
W_r, W_i, F_out, Reservoir_state_training, Output_training = \
    rc.train(N, D, Rou, Sigma, Alpha, Beta, Trajectory_training, plot=False,
             basis_function_1=bf.original, basis_function_2=bf.square,
             activation_function=Activation_function)

# Trajectory for Predicting
Predicting_time = int(222)
Time_predicting, Trajectory_predicting = data.lorenz(length=Predicting_time, sample=0.01, discard=0,
                                                     x0=list(Trajectory_training[-1, :]),
                                                     sigma=10, beta=8/3, rho=28)

# Self Predicting Process
Reservoir_state_predicting = np.zeros((Predicting_time, N))
Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]

Output_predicting = rc.self_predict(W_r, W_i, F_out, Trajectory_predicting, Reservoir_state_predicting,
                                    activation_function=Activation_function, plot=True)

# Valuation
Distance, RMSE, NRMSE, MAPE = rc.error_evaluate(Trajectory_predicting, Output_predicting,
                                                Time_predicting * 8.93203108e-01, plot=True)
