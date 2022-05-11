import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')

# Trajectory
Function_trajectory = rc.sprott

# Capacity
Capacity_training = 5000
Capacity_predicting = 2000

# Parameters
N = 1000
D = 3
Beta = 1e-4
Sigma = 1

# Function
Function_activation = rc.elu
Function_basis_1 = rc.original
Function_basis_2 = rc.square

# Path
Path = f'Result/Activation/{Function_activation.__name__}'
try:
    os.makedirs(Path + '/pic')
except FileExistsError:
    pass

Result = pd.DataFrame()
Rou_list = np.linspace(0, 1, 21)[1:]
for Rou in tqdm(Rou_list):
    
    for Times in range(20):
        # Trajectory for Training
        Start_pos = list(np.random.rand(D))
        Time_training, Trajectory_training = \
            Function_trajectory(length=Capacity_training, sample=0.01, 
                                x0=Start_pos, discard=0)

        # Train Process
        W_r, W_i, F_out, Reservoir_state_training, Output_training = \
            rc.train(N, D, Rou, Sigma, Beta, Trajectory_training, plot=False,
                     basis_function_1=Function_basis_1, 
                     basis_function_2=Function_basis_2,
                     activation_function=Function_activation)

        # Trajectory for Predicting
        Time_predicting, Trajectory_predicting = \
            Function_trajectory(length=Capacity_predicting, sample=0.01, 
                                discard=0, x0=list(Trajectory_training[-1, :]))

        # Self Predicting Process
        Reservoir_state_predicting = np.zeros((Capacity_predicting, N))
        Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]

        Output_predicting = \
            rc.predict(W_r, W_i, F_out, Trajectory_predicting, 
                       Reservoir_state_predicting,
                       activation_function=Function_activation, 
                       plot=True, save_path=Path + '/pic' + 
                       f'/predict_{str(Rou)[:4]}_{Times}.svg')

        # Valuation
        _, Evaluation = \
            rc.error_evaluate(Trajectory_predicting, Output_predicting,
                              Time_predicting, plot=True, 
                              save_path=Path + '/pic' + 
                              f'/evaluation_{str(Rou)[:4]}_{Times}.svg')

        Result = Result.append(pd.DataFrame(Evaluation, index=[str(Rou)[:4]]))

Result.to_csv(Path + f'/result_{Function_activation.__name__}.csv')

plt.figure(figsize=(10, 5))
Result_median = pd.DataFrame(np.zeros((len(Rou_list), Result.shape[1])), 
                             columns=Result.columns, 
                             index=[str(i)[:4] for i in Rou_list])
for indicator in Result.columns:
    for Rou in Rou_list:
        median = Result.loc[str(Rou)[:4]][indicator]
        median = np.median(median[~np.isnan(median)])
        Result_median.loc[str(Rou)[:4]][indicator] = median
    print(Result_median.index[np.argmin(Result_median[indicator])])
    plt.plot(Result_median[indicator], label=indicator)
    plt.legend()
plt.savefig(Path + '/Evaluation.svg', format='svg')

Result_median.to_csv(Path + 
                     f'/result_median_{Function_activation.__name__}.csv')
