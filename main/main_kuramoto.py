import numpy as np
import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')

# Trajectory
Function_trajectory = rc.kuramoto

# Capacity
Discard = 2000
Capacity_training = 5000
Capacity_predicting = 1000

# Parameters
N = 1000
D = 10
Beta = 1e-4
Sigma = 1
Rou = 0.1

# Function
Function_activation = rc.soft_plus
Function_basis_1 = rc.sin
Function_basis_2 = rc.cos

# Training Process
Start_pos = list(2 * np.pi * np.random.rand(int(D)))
Omega = np.ones(D)
Omega[:2] = 0.9
Omega[2:5] = 0.6
Omega[5:] = 0.4
K = 0.5

Time, Trajectory = \
    Function_trajectory(length=Capacity_training + Capacity_predicting, 
                        sample=0.01, x0=Start_pos, omega=Omega, k=K, 
                        discard=Discard)
    
Time_training, Trajectory_training = \
    (Time[:(Capacity_training - Discard)], 
     Trajectory[:(Capacity_training - Discard), :])
Time_predicting, Trajectory_predicting = \
    (Time[(Capacity_training - Discard):] - Time[(Capacity_training - Discard)], 
     Trajectory[(Capacity_training - Discard):, :])

W_r, W_i, F_out, Reservoir_state_training, Output_training = \
    rc.train(N, D, Rou, Sigma, Beta, Trajectory_training, plot=True,
             basis_function_1=Function_basis_1, 
             basis_function_2=Function_basis_2,
             activation_function=Function_activation)

# Predicting Process
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
