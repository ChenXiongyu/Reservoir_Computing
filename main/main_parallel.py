import numpy as np
import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')


# Capacity
Capacity_training = 5000
Capacity_predicting = 1500    

# Parameters
N = [500, 500]
D = 3
Beta = 1e-4
Sigma = [1, 1]
Rou = [0.1, 0.1]

# Function
Function_activation = [rc.soft_plus, rc.soft_plus]
Function_basis_1 = rc.original
Function_basis_2 = rc.square

# Trajectory
Function_trajectory = rc.sprott
Start_pos = list(np.random.rand(int(D)))
Time, Trajectory = \
    Function_trajectory(length=Capacity_training + Capacity_predicting, 
                        sample=0.01, x0=Start_pos)
Time_training, Trajectory_training = \
    (Time[:Capacity_training], 
     Trajectory[:Capacity_training, :])
Time_predicting, Trajectory_predicting = \
    (Time[Capacity_training:] - Time[Capacity_training], 
     Trajectory[Capacity_training:, :])

# Training
W_r, W_i, F_out, Reservoir_state_training, Output_training = \
    rc.train_parallel(N, D, Rou, Sigma, Beta, Trajectory_training, Function_activation, 
                      function_basis_1=Function_basis_1, function_basis_2=Function_basis_2, 
                      plot=True)

# Predicting Process
Reservoir_state_predicting = []
for R in range(len(N)):
    N_r = N[R]
    Reservoir_state_predicting_r = np.zeros((Capacity_predicting, N_r))
    Reservoir_state_predicting_r[0, :] = Reservoir_state_training[R][-1, :]
    Reservoir_state_predicting.append(Reservoir_state_predicting_r)


Output_predicting = \
    rc.predict_parallel(W_r, W_i, F_out, Trajectory_predicting, Reservoir_state_predicting, 
                        Function_activation, plot=True)

# Valuation
Distance, Evaluation = rc.error_evaluate(Trajectory_predicting, 
                                         Output_predicting,
                                         Time_predicting, plot=True)
