import numpy as np
import pandas as pd
from tqdm import tqdm
import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')

Result = pd.DataFrame()
for Times in tqdm(range(10)):
    # Trajectory
    Function_trajectory = rc.lorenz

    # Capacity
    Capacity_training = 5555
    Capacity_predicting = 222

    # Parameters
    N = [500, 500, 500, 500]

    D = 3
    Beta = 1e-4

    Sigma = [1, 1, 1, 1]

    Rou = [0.75, 0.75, 0.75, 0.75]

    # Function
    Function_activation = [rc.soft_plus, rc.soft_plus, rc.soft_plus, rc.soft_plus]
    Function_basis_1 = rc.original
    Function_basis_2 = rc.square

    # Training Process
    Start_pos = list(np.random.rand(int(D)))
    Time_training, Trajectory_training = \
        Function_trajectory(length=Capacity_training, sample=0.01, 
                            x0=Start_pos, discard=0)

    W_r, W_i, F_out, Reservoir_state_training, Output_training = \
        rc.train_parallel(N, D, Rou, Sigma, Beta, Trajectory_training, 
                        Function_activation, 
                        function_basis_1=Function_basis_1, 
                        function_basis_2=Function_basis_2, 
                        plot=False)

    # Predicting Process
    Time_predicting, Trajectory_predicting = \
        Function_trajectory(length=Capacity_predicting, 
                            sample=0.01, discard=0, 
                            x0=list(Trajectory_training[-1, :]))

    Reservoir_state_predicting = []
    for R in range(len(N)):
        N_r = N[R]
        Reservoir_state_predicting_r = np.zeros((Capacity_predicting, N_r))
        Reservoir_state_predicting_r[0, :] = Reservoir_state_training[R][-1, :]
        Reservoir_state_predicting.append(Reservoir_state_predicting_r)

    Output_predicting = \
        rc.predict_parallel(W_r, W_i, F_out, Trajectory_predicting, Reservoir_state_predicting, 
                            Function_activation, plot=False)

    # Valuation
    Distance, Evaluation = rc.error_evaluate(Trajectory_predicting, 
                                            Output_predicting,
                                            Time_predicting, plot=True)
    Result = Result.append(pd.DataFrame(Evaluation, index=[Times]))
    
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(Trajectory_predicting[:, 0], Trajectory_predicting[:, 1], Trajectory_predicting[:, 2], c='r')
# ax.plot(Output_predicting[:, 0], Output_predicting[:, 1], Output_predicting[:, 2], c='b', ls='--')
# plt.savefig('Sprott.svg', format='svg')
