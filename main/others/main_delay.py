import numpy as np
from nolitsa import data

import basis_function as bf
import reservoir_computing as rc
import activation_function as ac

import warnings
warnings.filterwarnings('ignore')

# Parameters
N = 1000
D = 3
Alpha = 0.27
Beta = 1e-4
Sigma = 1
Rou = 0.15

Delay = 1

# Activation Function
Activation_function = ac.relu

# Trajectory for Training
Start_pos = list(np.random.rand(3))
Time_training, Trajectory_training = data.lorenz(length=5555, sample=0.01, x0=Start_pos, discard=0,
                                                 sigma=10, beta=8/3, rho=28)

# Train Process
W_r, W_i, F_out, Reservoir_state_training, Output_training = \
    rc.train_delay(N, D, Rou, Sigma, Alpha, Beta, Delay, Trajectory_training, plot=False,
                   basis_function_1=bf.original, basis_function_2=bf.square,
                   activation_function=Activation_function)

# Trajectory for Predicting
Predicting_time = int(222)
Time_predicting, Trajectory_predicting = data.lorenz(length=Predicting_time, sample=0.01, discard=0,
                                                     x0=list(Trajectory_training[-1, :]),
                                                     sigma=10, beta=8/3, rho=28)

# Self Predicting Process
Reservoir_state_predicting = np.zeros((Predicting_time, N))
Reservoir_state_predicting[0:(Delay + 1), :] = Reservoir_state_training[-(Delay + 1), :]

Output_predicting = rc.self_predict_delay(W_r, W_i, F_out, Delay, Trajectory_predicting, Reservoir_state_predicting,
                                          activation_function=Activation_function, plot=True)

# Valuation
# lle_traj_training = rc.lle_lorenz(Trajectory_training)
lle_traj_training = 0.9
Distance, RMSE, NRMSE, MAPE = rc.error_evaluate(Trajectory_predicting, Output_predicting,
                                                Time_predicting * lle_traj_training, plot=True)
