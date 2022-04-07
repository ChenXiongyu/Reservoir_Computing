import numpy as np
import pandas as pd
from nolitsa import data
from tqdm import tqdm

import reservoir_computing as rc
import activation_function as ac

# Parameters
N = 1200
D = 3
Rou = 0.15
Sigma = 0.1
Alpha = 0.27
Beta = 1e-4

# Activation Function
Activation_function = ac.relu

# Trajectory for Training
Training_time = int(5555)
Start_pos_training_1 = list(np.random.rand(3))
Time_training_1, Trajectory_training_1 = data.lorenz(length=Training_time, sample=0.01, x0=Start_pos_training_1,
                                                     discard=0, sigma=10, beta=8/3, rho=28)
Start_pos_training_2 = list(np.random.rand(3))
Time_training_2, Trajectory_training_2 = data.lorenz(length=Training_time, sample=0.01, x0=Start_pos_training_2,
                                                     discard=0, sigma=10, beta=8/3, rho=28)

# Train Process
W_r_1, W_i_1, F_out_1, Reservoir_state_training_1, Output_training_1 = \
    rc.train(N, D, Rou, Sigma, Alpha, Beta, Trajectory_training_1,
             activation_function=Activation_function, plot=False)
W_r_2, W_i_2, F_out_2, Reservoir_state_training_2, Output_training_2 = \
    rc.train(N, D, Rou, Sigma, Alpha, Beta, Trajectory_training_2,
             activation_function=Activation_function, plot=False)

# Trajectory for Predicting
Predicting_time = int(2222)
Start_pos_predicting_1 = list(Trajectory_training_1[-2, :])
Time_predicting_1, Trajectory_predicting_1 = \
    data.lorenz(length=Predicting_time, sample=0.01, x0=Start_pos_predicting_1, discard=0, sigma=10, beta=8/3, rho=28)
Start_pos_predicting_2 = list(Trajectory_training_2[-2, :])
Time_predicting_2, Trajectory_predicting_2 = \
    data.lorenz(length=Predicting_time, sample=0.01, x0=Start_pos_predicting_2, discard=0, sigma=10, beta=8/3, rho=28)

# Coupled Predicting Process
Coupled_strength_list = np.linspace(0, 1, 11)[1:]
Noise_strength_list = np.linspace(0, 1, 11)[1:]

Result_rmse = pd.DataFrame()
Result_nrmse = pd.DataFrame()
Result_mape = pd.DataFrame()
for Coupled_strength in tqdm(Coupled_strength_list):
    RMSE_list = {}
    NRMSE_list = {}
    MAPE_list = {}
    for Noise_strength in Noise_strength_list:
        Reservoir_state_predicting_1 = np.zeros((Predicting_time, N))
        Reservoir_state_predicting_1[0, :] = Reservoir_state_training_1[-1, :]
        Reservoir_state_predicting_2 = np.zeros((Predicting_time, N))
        Reservoir_state_predicting_2[0, :] = Reservoir_state_training_2[-1, :]

        Output_predicting_1, Output_predicting_2 = \
            rc.coupled_predict(W_r_1, W_i_1, F_out_1, Reservoir_state_predicting_1, Trajectory_predicting_1,
                               W_r_2, W_i_2, F_out_2, Reservoir_state_predicting_2, Trajectory_predicting_2,
                               Coupled_strength, Noise_strength, activation_function=Activation_function)

        # Distance_1, RMSE_1, NRMSE_1, MAPE_1 = rc.error_evaluate(Trajectory_predicting_1, Output_predicting_1,
        #                                                         Time_predicting_1 * 8.93203108e-01, plot=False)
        # Distance_2, RMSE_2, NRMSE_2, MAPE_2 = rc.error_evaluate(Trajectory_predicting_2, Output_predicting_2,
        #                                                         Time_predicting_2 * 8.93203108e-01, plot=False)

        Distance, RMSE, NRMSE, MAPE = rc.error_evaluate(Output_predicting_1, Output_predicting_2,
                                                        Time_predicting_2 * 8.93203108e-01, time_start=222, plot=False)
        # rc.plot_trajectory(Output_predicting_1[222:, :], Output_predicting_2[222:, :])
        RMSE_list[Noise_strength] = RMSE
        NRMSE_list[Noise_strength] = NRMSE
        MAPE_list[Noise_strength] = MAPE

    result_rmse = pd.DataFrame(RMSE_list, index=[Coupled_strength])
    result_nrmse = pd.DataFrame(NRMSE_list, index=[Coupled_strength])
    result_mape = pd.DataFrame(MAPE_list, index=[Coupled_strength])
    Result_rmse = Result_rmse.append(result_rmse)
    Result_nrmse = Result_nrmse.append(result_nrmse)
    Result_mape = Result_mape.append(result_mape)

Path = 'Result'
Result_rmse.to_csv(Path + '/result_rmse.csv')
Result_nrmse.to_csv(Path + '/result_nrmse.csv')
Result_mape.to_csv(Path + '/result_mape.csv')
