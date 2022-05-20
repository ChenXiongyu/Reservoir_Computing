import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')


# Capacity
Capacity_training = 5000
Capacity_predicting = 2000

# Function
Function_activation = [rc.soft_plus, rc.soft_plus, rc.soft_plus, rc.soft_plus, rc.soft_plus]
Function_basis_1 = [rc.sin, rc.sin, rc.sin, rc.sin, rc.sin]
Function_basis_2 = [rc.cos, rc.cos, rc.cos, rc.cos, rc.cos]
Function_trajectory = rc.kuramoto
    
# Parameters
N = [1000, 1000, 1000, 1000, 1000]
D = 10
D_unobservable = 7
Beta = 1e-4 * np.ones(len(N))
Sigma = np.ones(len(N))
Rou = [0.9, 0.9, 0.9, 0.9, 0.9]
Noise_strength = 0

Result_median = pd.DataFrame()
for W in tqdm(list(range(10, -1, -1))):
    Evaluation = pd.DataFrame()
    for Times in range(5):
        Coupled_weights = np.random.rand(len(N))
        Coupled_weights[0] = W * 0.1
        Coupled_weights[1:] = \
            Coupled_weights[1:] / sum(Coupled_weights[1:]) * (1 - W * 0.1)

        W_r = []
        W_i = []
        F_out = []
        Reservoir_state_training = []
        Trajectory_predicting = []
            
        for R in range(len(N)):
            
            # Trajectory
            Start_pos = list(2 * np.pi * np.random.rand(int(D)))
            Omega = np.ones(D)
            Omega[:2] = 0.9
            Omega[2:5] = 0.6
            Omega[5:8] = 0.4
            Omega[8:] = 0.1
            K = 1

            Time, Trajectory = \
                Function_trajectory(length=Capacity_training + Capacity_predicting, 
                                    sample=0.01, x0=Start_pos, omega=Omega, k=K)
            Trajectory = np.sin(Trajectory)
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
                        activation_function=Function_activation[R], 
                        unobservable=D_unobservable)
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
                            reversed_weights=False)

        # Valuation
        Distance = np.zeros((Capacity_predicting, len(N)))
        for R in range(len(N)):
            Distance[:, R], Evaluation_r = rc.error_evaluate(Output_coupled, Output_predicting[R],
                                                            0, plot=False)
            Evaluation = Evaluation.append(pd.DataFrame(Evaluation_r, index=[R]))

    # Distance_sum = np.sum(Distance, axis=1)
    # Steps = 100
    # plt.figure()
    # plt.title(f'Synchronization after {Steps} Iteration Steps')
    # plt.plot(Distance_sum[Steps:])  # 迭代100步后的结果

    # plt.figure()
    # _ = plt.plot(Output_coupled, c='r', ls='--')
    # _ = plt.plot(Output_predicting[0], c='b', ls='--')
        
    result = {}
    for indicator in Evaluation.columns:
        result[indicator] = np.median(Evaluation.loc[0][indicator])
    Result_median = Result_median.append(pd.DataFrame(result, index=[W]))

print(Result_median)
Result_median.to_csv('coupled.csv')

plt.figure()
_ = plt.plot(Output_coupled, c='r', ls='--')
_ = plt.plot(Output_predicting[0], c='b', ls='--')


# Possible Trajectory after Synchronization
# Start_pos = list(2 * np.pi * np.random.rand(int(D)))
# _, Trajectory_coupled = \
#     Function_trajectory(length=Capacity_predicting - Steps + 1000, 
#                         sample=0.01, x0=Start_pos, omega=Omega, k=K, discard=1000)
# Trajectory_coupled = np.sin(Trajectory_coupled)

# Distance_coupled, Evaluation_coupled = \
#     rc.error_evaluate(Output_coupled[Steps:], Trajectory_coupled, 0, plot=True)
# Evaluation = Evaluation.append(pd.DataFrame(Evaluation_coupled, index=['coupled']))

# plt.figure()
# _ = plt.plot(Trajectory_coupled, c='r')
        

