import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')


# Capacity
Capacity_training = 5000
Capacity_predicting = 2000

# Parameters
N = [1000, 1000, 1000, 1000, 1000]
D = 3
Beta = 1e-4 * np.ones(len(N))
Sigma = np.ones(len(N))
Rou = [0.1, 0.1, 0.1, 0.1, 0.1]
Coupled_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
Noise_strength = 0

# Function
Function_activation = [rc.soft_plus, rc.relu, rc.prelu, rc.elu, rc.tanh]
Function_basis_1 = [rc.original, rc.original, rc.original, rc.original, rc.original]
Function_basis_2 = [rc.square, rc.square, rc.square, rc.square, rc.square]
Function_trajectory = rc.lorenz


W_r = []
W_i = []
F_out = []
Reservoir_state_training = []
Trajectory_predicting = []
    
for R in range(len(N)):
    
    # Trajectory
    Start_pos = list(np.random.rand(int(D)))
    Time, Trajectory = \
        Function_trajectory(length=Capacity_training + Capacity_predicting, 
                            sample=0.01, x0=Start_pos)
    _, Trajectory_training_r = \
        (Time[:Capacity_training], 
        Trajectory[:Capacity_training, :])
    _, Trajectory_predicting_r = \
        (Time[Capacity_training:] - Time[Capacity_training], 
        Trajectory[Capacity_training:, :])

    # Training
    W_r_r, W_i_r, F_out_r, Reservoir_state_training_r, Output_training_r = \
        rc.train(N[R], D, Rou[R], Sigma[R], Beta[R], Trajectory_training_r, plot=False,
                basis_function_1=Function_basis_1[R], 
                basis_function_2=Function_basis_2[R],
                activation_function=Function_activation[R])
    W_r.append(W_r_r)
    W_i.append(W_i_r)
    F_out.append(F_out_r)
    Reservoir_state_training.append(Reservoir_state_training_r)
    Trajectory_predicting.append(Trajectory_predicting_r)

Reservoir_state_predicting = []
for R in range(len(N)):
    Reservoir_state_predicting_r = np.zeros((Capacity_predicting, N[R]))
    Reservoir_state_predicting_r[0, :] = Reservoir_state_training[R][-1, :]
    Reservoir_state_predicting.append(Reservoir_state_predicting_r)
    
# Predicting
Output_coupled, Output_predicting = \
    rc.predict_coupled(W_r, W_i, Reservoir_state_predicting, Trajectory_predicting,
                       F_out, Function_activation, Coupled_weights, Noise_strength, 
                       reversed_weights=True)

# Valuation
Evaluation = pd.DataFrame()
Distance = np.zeros((Capacity_predicting, len(N)))
for R in range(len(N)):
    Distance[:, R], Evaluation_r = rc.error_evaluate(Output_coupled, Output_predicting[R],
                                                     0, plot=False)
    Evaluation = Evaluation.append(pd.DataFrame(Evaluation_r, index=[R]))

Distance_sum = np.sum(Distance, axis=1)
print(Evaluation)

# Possible Trajectory after Synchronization
Steps = 100

plt.figure()
plt.title(f'Synchronization after {Steps} Iteration Steps')
plt.plot(Distance_sum[Steps:])  # 迭代100步后的结果

Start_pos = list(Output_coupled[Steps, :])
_, Trajectory_coupled = \
    Function_trajectory(length=Capacity_predicting - Steps, 
                        sample=0.01, x0=Start_pos)
Distance_coupled, Evaluation_coupled = \
    rc.error_evaluate(Output_coupled[Steps:], Trajectory_coupled, 0, plot=True)
Evaluation = Evaluation.append(pd.DataFrame(Evaluation_coupled, index=['coupled']))
rc.plot_trajectory(Trajectory_coupled, Output_coupled)