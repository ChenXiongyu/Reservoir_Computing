import numpy as np
import pandas as pd
from tqdm import tqdm
# import matplotlib.pyplot as plt

from sklearn import linear_model
import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')


# Capacity
N = 1000
Capacity = {'training': 5000, 'predicting': 100, 'coupling': 1500}


Ranking = pd.DataFrame()
for Times in tqdm(range(5)):
    while True:
        # Trajectory
        D = 3
        Function_trajectory = rc.roessler
        Start_pos = list(np.random.rand(int(D)))
        _, Trajectory_training = \
            Function_trajectory(length=Capacity['training'], 
                                sample=0.01, x0=Start_pos, discard=0)
        _, Trajectory_predicting = \
            Function_trajectory(length=Capacity['predicting'], 
                                sample=0.01, discard=0,
                                x0=list(Trajectory_training[-1, :]))
        _, Trajectory_coupling = \
            Function_trajectory(length=Capacity['coupling'], 
                                sample=0.01, discard=0,
                                x0=list(Trajectory_predicting[-1, :]))


        # Function
        Function_list = [rc.elu, rc.soft_plus, rc.prelu, 
                        rc.relu, rc.tanh, rc.sigmoid]
        Evaluation = pd.DataFrame()
        Distance = np.zeros((Capacity['coupling'], len(Function_list) + 2))


        # Training
        X_predict = np.zeros((Capacity['predicting'], len(Function_list), D))
        Parameter = pd.read_csv('Parameter.csv', index_col=0)
        W_r, W_i, F_out, Reservoir_state_end = {}, {}, {}, {}

        for i in range(len(Function_list)):
            Function_activation = Function_list[i]
            Function_name = Function_activation.__name__
            Rou = Parameter.loc[Function_trajectory.__name__][Function_name]
            while True:
                (Output_predicting, Trajectory_predicting, Time_predicting, 
                w_r, w_i, f_out, reservoir_state_end)= \
                    rc.rc_model(Capacity, Function_trajectory, Start_pos, Rou, 
                                Function_activation)
                if np.sum(np.isnan(Output_predicting) | 
                            np.isinf(Output_predicting)) == 0:
                    break
                
            W_r[Function_name] = w_r
            W_i[Function_name] = w_i
            F_out[Function_name] = f_out
            Reservoir_state_end[Function_name] = reservoir_state_end
            
            for d in range(D):
                X_predict[:, i, d] = Output_predicting[:, d]
        y_predict = Trajectory_predicting
        Reservoir_state_continue = Reservoir_state_end.copy()


        # Regression
        regr = linear_model.Ridge(alpha=5)
        X_predict = np.vstack((X_predict[:, :, 0], 
                            X_predict[:, :, 1], 
                            X_predict[:, :, 2]))
        y_predict = np.hstack((y_predict[:, 0], 
                            y_predict[:, 1],
                            y_predict[:, 2]))
        regr.fit(X_predict, y_predict)


        # Coupling
        output_regression = np.zeros((Trajectory_coupling.shape[0], 
                                    Trajectory_coupling.shape[1], 
                                    len(Function_list)))
        for j in range(len(Function_list)):
            Output_continue = np.zeros(Trajectory_coupling.shape)
            Output_continue[0, :] = Trajectory_coupling[0, :]
            Function_activation = Function_list[j]
            Function_name = Function_activation.__name__
            for i in range(1, len(Trajectory_coupling)):
                Reservoir_state_continue[Function_name] = \
                    Function_activation(
                        np.dot(W_r[Function_name], 
                            Reservoir_state_continue[Function_name]) + 
                        np.dot(W_i[Function_name], 
                            Output_continue[i - 1, :]))
                Output_continue[i, :] = F_out[Function_name](
                    Reservoir_state_continue[Function_name])
            output_regression[:, :, j] = Output_continue
            # rc.plot_trajectory(Trajectory_coupling, Output_continue)
            distance, evaluation = \
                rc.error_evaluate(Trajectory_coupling, Output_continue, 
                                0, plot=False)
            Evaluation = Evaluation.append(
                pd.DataFrame(evaluation, index=[Function_name]))
            Distance[:, j] = distance
        
        Output_regression = np.zeros(Trajectory_coupling.shape)
        try:
            for d in range(D):
                Output_regression[:, d] = \
                    regr.predict(output_regression[:, d, :])
        except ValueError:
            continue
        distance, evaluation = \
            rc.error_evaluate(Trajectory_coupling, Output_regression, 
                            0, plot=False)
        Evaluation = Evaluation.append(
            pd.DataFrame(evaluation, index=['regression']))
        Distance[:, -2] = distance

        Output_coupling = np.zeros(Trajectory_coupling.shape)
        Output_coupling[0, :] = Trajectory_coupling[0, :]
        X_coupling = np.zeros((D, len(Function_list)))
        try:
            for i in range(1, len(Trajectory_coupling)):
                for j in range(len(Function_list)):
                    Function_activation = Function_list[j]
                    Function_name = Function_activation.__name__
                    Reservoir_state_end[Function_name] = \
                        Function_activation(
                            np.dot(W_r[Function_name], 
                                Reservoir_state_end[Function_name]) + 
                            np.dot(W_i[Function_name], 
                                Output_coupling[i - 1, :]))
                    X_coupling[:, j] = F_out[Function_name](
                        Reservoir_state_end[Function_name])
                Output_coupling[i, :] = regr.predict(X_coupling)
        except ValueError:
            continue
        # rc.plot_trajectory(Trajectory_coupling, Output_coupling)
        distance, evaluation = \
            rc.error_evaluate(Trajectory_coupling, Output_coupling, 
                            0, plot=False)
        Evaluation = Evaluation.append(
            pd.DataFrame(evaluation, index=['coupling']))
        Distance[:, -1] = distance

        # print(Evaluation)
        # plt.figure()
        # plt.plot(Distance, label=Evaluation.index)
        # plt.legend()
        # plt.savefig('Distance.svg', format='svg')

        # Rankings
        ranking = {}
        RMSE = Evaluation['RMSE'].values
        for f in Evaluation.index:
            ranking[f] = \
                np.where(Evaluation['RMSE'].index
                        [np.argsort(RMSE)] == f)[0][0]
        ranking = pd.DataFrame(ranking, index=[Times])
        Ranking = Ranking.append(ranking)
        
        break

# Ranking.to_csv('ranking.csv')
print(Ranking)
