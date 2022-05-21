import numpy as np
import pandas as pd
from tqdm import tqdm


import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')

Result_w = pd.DataFrame()
for W in tqdm(range(11)):
    
    Noise_strength = W * 0.1
    
    # Parameters
    N = [1000, 1000]
    Rou = [0.75, 0.75]
    Coupled_weights = np.ones(len(N)) / len(N)
    Function_activation = [rc.soft_plus, rc.soft_plus]
    Function_basis_1 = [rc.original, rc.original]
    Function_basis_2 = [rc.square, rc.square]
    Reverse = False

    D = 3
    Beta = 1e-4 * np.ones(len(N))
    Sigma = np.ones(len(N))

    # Capacity
    Capacity_training = 5000
    Capacity_predicting = 5000

    # Function
    Function_trajectory = rc.lorenz

    Result = pd.DataFrame()
    for Times in range(5):
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
                            reversed_weights=Reverse)

        # Valuation
        Evaluation = pd.DataFrame()
        Distance = np.zeros((Capacity_predicting, len(N)))
        for R in range(len(N)):
            Distance[:, R], Evaluation_r = rc.error_evaluate(Output_coupled, Output_predicting[R],
                                                            0, plot=False)
            Evaluation_r['Max'] = np.max(Distance[100:, R])
            Evaluation = Evaluation.append(pd.DataFrame(Evaluation_r, index=[R]))

        # rc.plot_trajectory(Output_coupled, Output_predicting[0])
        Result = Result.append(Evaluation)

    # print(Result)
    Result_median = pd.DataFrame()
    for R in range(len(N)):
        result = {}
        for indicator in Result.columns:
            r = Result.loc[R][indicator]
            r = r[~np.isnan(r)]
            result[indicator] = np.median(r)
        result['SC'] = 1 - np.sum(np.isnan(Result['RMSE'])) / Result.shape[0]
        Result_median = Result_median.append(pd.DataFrame(result, index=[R]))
    result = np.sum(Result_median)
    result.name = W
    Result_w = Result_w.append(result)
print(Result_w)
Result_w['SC'] = Result_w['SC'] / 2
Result_w.to_csv('coupled.csv')

import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


fig, ax1 = plt.subplots()
plt.xticks(np.linspace(0, 10, 11), [str(i)[:3] for i in np.linspace(0, 1, 11)])

ax1.plot(Result_w['RMSE'], label='RMSE', c='r')
ax1.set_xlabel("Noise Strength")
ax1.set_ylabel("RMSE")
plt.legend()

ax2 = ax1.twinx()
ax2.plot(Result_w['SC'], label='SC', c='b')
ax2.set_xlabel("Noise Strength")
ax2.set_ylabel("SC")
plt.legend()

plt.savefig('noise.svg')
