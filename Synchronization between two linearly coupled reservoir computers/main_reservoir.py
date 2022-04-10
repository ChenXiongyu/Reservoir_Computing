import numpy as np
import pandas as pd
from tqdm import tqdm
from nolitsa import data
import matplotlib.pyplot as plt

import basis_function as bf
import reservoir_computing as rc
import activation_function as ac

# Parameters
N = 1000
D = 3
Alpha = 0.27
Beta = 1e-4
Sigma = 1
Rou = 0.15
Probability = 0.0015

# Activation Function
Activation_function = ac.relu

Result = pd.DataFrame()
RMSE_dict, NRMSE_dict, MAPE_dict = {}, {}, {}
Antisymmetry_list = np.linspace(0, 1, 11)
for Antisymmetry in tqdm(Antisymmetry_list):
    Symmetry = 1 - Antisymmetry

    RMSE_list, NRMSE_list, MAPE_list = [], [], []
    for Times in range(1):
        # Trajectory for Training
        Start_pos = list(np.random.rand(3))
        Time_training, Trajectory_training = data.lorenz(length=5555, sample=0.01, x0=Start_pos, discard=0,
                                                         sigma=10, beta=8/3, rho=28)

        # Train Process
        W_r, W_i, F_out, Reservoir_state_training, Output_training = \
            rc.train_reservoir(N, Rou, Sigma, Alpha, Beta, Probability, Symmetry, Antisymmetry,
                               Trajectory_training, plot=False,
                               basis_function_1=bf.original, basis_function_2=bf.square,
                               activation_function=Activation_function)

        # Trajectory for Predicting
        Predicting_time = int(555)
        Time_predicting, Trajectory_predicting = data.lorenz(length=Predicting_time, sample=0.01, discard=0,
                                                             x0=list(Trajectory_training[-1, :]),
                                                             sigma=10, beta=8/3, rho=28)

        # Self Predicting Process
        Reservoir_state_predicting = np.zeros((Predicting_time, N))
        Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]

        Output_predicting = rc.self_predict(W_r, W_i, F_out, Trajectory_predicting, Reservoir_state_predicting,
                                            activation_function=Activation_function, plot=False)

        # Valuation
        # lle_traj_training = rc.lle_lorenz(Trajectory_training)
        lle_traj_training = 0.9
        Distance, RMSE, NRMSE, MAPE = rc.error_evaluate(Trajectory_predicting, Output_predicting,
                                                        Time_predicting * lle_traj_training, plot=False)
        RMSE_list.append(RMSE)
        NRMSE_list.append(NRMSE)
        MAPE_list.append(MAPE)

    RMSE_dict[int(Probability * 10000)] = RMSE_list
    NRMSE_dict[int(Probability * 10000)] = NRMSE_list
    MAPE_dict[int(Probability * 10000)] = MAPE_list

    result = pd.DataFrame({'RMSE': np.median(RMSE_list), 'NRMSE': np.median(NRMSE_list), 'MAPE': np.median(MAPE_list)},
                          index=[str(Antisymmetry)[:4]])
    # print(result)
    Result = Result.append(result)

RMSE_dict = pd.DataFrame(RMSE_dict)
NRMSE_dict = pd.DataFrame(NRMSE_dict)
MAPE_dict = pd.DataFrame(MAPE_dict)

Path = 'Result'
Result.to_csv(Path + '/result_symmetry_antisymmetry.csv')

for indicator in range(3):
    plt.subplot(3, 1, indicator + 1)
    plt.plot(Result[Result.columns[indicator]], label=Result.columns[indicator])
    plt.legend()
