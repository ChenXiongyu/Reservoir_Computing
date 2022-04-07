import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from nolitsa import data
import matplotlib.pyplot as plt

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

Result = pd.DataFrame()
Rou_list = np.linspace(0, 1, 21)[1:]
for Rou in tqdm(Rou_list):

    RMSE_list, NRMSE_list, MAPE_list = [], [], []
    for Times in range(10):
        # Trajectory for Training
        Start_pos = list(np.random.rand(3))
        Time_training, Trajectory_training = data.lorenz(length=5555, sample=0.01, x0=Start_pos, discard=0,
                                                         sigma=10, beta=8/3, rho=28)

        # Train Process
        W_r, W_i, F_out, Reservoir_state_training, Output_training = \
            rc.train(N, D, Rou, Sigma, Alpha, Beta, Trajectory_training, plot=False,
                     basis_function_1=bf.original, basis_function_2=bf.original,
                     activation_function=ac.tanh)

        # Trajectory for Predicting
        Predicting_time = int(222)
        Time_predicting, Trajectory_predicting = data.lorenz(length=Predicting_time, sample=0.01, discard=0,
                                                             x0=list(Trajectory_training[-1, :]),
                                                             sigma=10, beta=8/3, rho=28)

        # Self Predicting Process
        Reservoir_state_predicting = np.zeros((Predicting_time, N))
        Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]

        Output_predicting = rc.self_predict(W_r, W_i, F_out, Trajectory_predicting, Reservoir_state_predicting,
                                            activation_function=ac.tanh, plot=False)

        # Valuation
        Distance, RMSE, NRMSE, MAPE = rc.error_evaluate(Trajectory_predicting, Output_predicting,
                                                        Time_predicting * 8.93203108e-01, plot=False)
        RMSE_list.append(RMSE)
        NRMSE_list.append(NRMSE)
        MAPE_list.append(MAPE)

    result = pd.DataFrame({'RMSE': np.median(RMSE_list), 'NRMSE': np.median(NRMSE_list), 'MAPE': np.median(MAPE_list)},
                          index=[str(Rou)[:4]])
    # print(result)
    Result = Result.append(result)

Path = 'Result'
try:
    os.makedirs(Path)
except FileExistsError:
    pass
Result.to_csv(Path + '/result_linear_tanh.csv')

for indicator in Result.columns:
    plt.plot(Result[indicator], label=indicator)
plt.legend()
