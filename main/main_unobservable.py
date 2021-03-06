import numpy as np
import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')

# Trajectory
Function_trajectory = rc.roessler

# Capacity
Capacity_training = 5000
Capacity_predicting = 1000

# Parameters
N = 1000
D = 3
D_unobservable = 1
Beta = 1e-4
Sigma = 1
Rou = 0.1

# Function
Function_activation = rc.soft_plus
Function_basis_1 = rc.original
Function_basis_2 = rc.original

# Training Process
Start_pos = list(np.random.rand(int(D)))
Time_training, Trajectory_training = \
    Function_trajectory(length=Capacity_training, sample=0.01, 
                        x0=Start_pos, discard=0)

W_r, W_i, F_out, Reservoir_state_training, Output_training = \
    rc.train(N, D, Rou, Sigma, Beta, Trajectory_training, plot=True,
             basis_function_1=Function_basis_1, 
             basis_function_2=Function_basis_2,
             activation_function=Function_activation, 
             unobservable=D_unobservable)

# Predicting Process
Time_predicting, Trajectory_predicting = \
    Function_trajectory(length=Capacity_predicting, 
                        sample=0.01, discard=0, 
                        x0=list(Trajectory_training[-1, :]))

Reservoir_state_predicting = np.zeros((Capacity_predicting, N))
Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]

Output_predicting = \
    rc.predict(W_r, W_i, F_out, Trajectory_predicting, 
               Reservoir_state_predicting,
               activation_function=Function_activation, 
               plot=True)

# Valuation
Distance, Evaluation = rc.error_evaluate(Trajectory_predicting, 
                                         Output_predicting,
                                         Time_predicting, plot=True)
