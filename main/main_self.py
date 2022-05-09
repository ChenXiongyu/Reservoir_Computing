import numpy as np
import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')

# Trajectory
Function_trajectory = rc.sprott

# Capacity
Capacity_training = 55555
Capacity_predicting = 1111

# Parameters
N = 1000
D= 3
Beta = 1e-4
Sigma = 1
Rou = 0.5

# Function
Function_activation = rc.tanh
Function_basis_1 = rc.original
Function_basis_2 = rc.square

# Training Process
Start_pos = list(np.random.rand(int(D)))
Time_training, Trajectory_training = \
    Function_trajectory(length=Capacity_training, sample=0.01, 
                        x0=Start_pos, discard=0)

W_r, W_i, F_out, Reservoir_state_training, Output_training = \
    rc.train(N, D, Rou, Sigma, Beta, Trajectory_training, plot=True,
             basis_function_1=Function_basis_1, 
             basis_function_2=Function_basis_2,
             activation_function=Function_activation)

# Predicting Process
Predicting_time = int(Capacity_predicting)
Time_predicting, Trajectory_predicting = \
    Function_trajectory(length=Predicting_time, sample=0.01, discard=0, 
                        x0=list(Trajectory_training[-1, :]))

Reservoir_state_predicting = np.zeros((Predicting_time, N))
Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]

Output_predicting = rc.self_predict(W_r, W_i, F_out, Trajectory_predicting, 
                                    Reservoir_state_predicting,
                                    activation_function=Function_activation, 
                                    plot=True)

# Valuation
Distance, RMSE, NRMSE, MAPE = rc.error_evaluate(Trajectory_predicting, 
                                                Output_predicting,
                                                Time_predicting, 
                                                plot=True)
