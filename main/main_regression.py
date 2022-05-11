import numpy as np
import pandas as pd
from tqdm import tqdm
# import matplotlib.pyplot as plt

from sklearn import linear_model
import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')

def rc_model(capacity, trajectory, start, rou, activation):
    capacity_training = capacity['training']
    capacity_predicting = capacity['predicting']
    
    # Parameters
    n = 1000
    d = len(start)

    beta = 1e-4
    sigma = 1

    # Function
    function_basis_1 = rc.original
    function_basis_2 = rc.square

    # Training Process
    _, trajectory_training = \
        trajectory(length=capacity_training, sample=0.01, 
                   x0=start, discard=0)

    w_r, w_i, f_out, reservoir_state_training, _ = \
        rc.train(n, d, rou, sigma, beta, trajectory_training, 
                 plot=False,
                 basis_function_1=function_basis_1, 
                 basis_function_2=function_basis_2,
                 activation_function=activation)

    # Predicting Process
    time_predicting, trajectory_predicting = \
        trajectory(length=capacity_predicting, 
                   sample=0.01, discard=0,
                   x0=list(trajectory_training[-1, :]))

    reservoir_state_predicting = np.zeros((capacity_predicting, n))
    reservoir_state_predicting[0, :] = reservoir_state_training[-1, :]

    output_predicting = \
        rc.predict(w_r, w_i, f_out, trajectory_predicting, 
                reservoir_state_predicting,
                activation_function=activation, 
                plot=False)
        
    return output_predicting, trajectory_predicting, time_predicting


Ranking = pd.DataFrame()
for Times in tqdm(range(5)):


    # Trajectory
    D = 3
    Function_trajectory = rc.roessler
    Start_pos = list(np.random.rand(int(D)))


    # Capacity
    Capacity = {'training': 5000, 'predicting': 3000}
    Regression = 2000


    # Function
    Function_list = [rc.elu, rc.soft_plus, rc.prelu, 
                     rc.relu, rc.tanh, rc.sigmoid]
    Evaluation = pd.DataFrame()
    Distance = np.zeros((Capacity['predicting'] - Regression, 
                        len(Function_list) + 1))


    # Train
    X_train = np.zeros((Regression, len(Function_list), D))
    X_predict = np.zeros((Capacity['predicting'] - Regression, 
                        len(Function_list), D))
    Parameter = pd.read_csv('Parameter.csv', index_col=0)
    for i in range(len(Function_list)):
        Function_activation = Function_list[i]
        Function_name = Function_activation.__name__
        Rou = Parameter.loc[Function_trajectory.__name__][Function_name]
        while True:
            Output_predicting, Trajectory_predicting, Time_predicting = \
                rc_model(Capacity, Function_trajectory, Start_pos, Rou, 
                        Function_activation)
            if np.sum(np.isnan(Output_predicting) | 
                      np.isinf(Output_predicting)) == 0:
                break
        for d in range(D):
            X_train[:, i, d] = Output_predicting[:Regression, d]
            X_predict[:, i, d] = Output_predicting[Regression:, d]
        distance, evaluation = \
            rc.error_evaluate(Trajectory_predicting[Regression:, :], 
                            X_predict[:, i, :], 0, 
                            plot=False)
        Distance[:, i] = distance
        Evaluation = Evaluation.append(
            pd.DataFrame(evaluation, index=[Function_name]))
    y_train = Trajectory_predicting[:Regression, :]
    y_predict = Trajectory_predicting[Regression:, :]


    # Regression
    regr = linear_model.Ridge(alpha=5)
    X_train = np.vstack((X_train[:, :, 0], 
                         X_train[:, :, 1], 
                         X_train[:, :, 2]))
    y_train = np.hstack((y_train[:, 0], y_train[:, 1], y_train[:, 2]))
    regr.fit(X_train, y_train)
    
    y = np.zeros(y_predict.shape)
    for d in range(D):
        y[:, d] = regr.predict(X_predict[:, :, d])

    # rc.plot_trajectory(y_predict, y)
    
    distance, evaluation = \
        rc.error_evaluate(y_predict, y, 0, 
                        plot=False)
    Distance[:, -1] = distance
    Evaluation = Evaluation.append(
        pd.DataFrame(evaluation, 
                        index=['regression']))


    # # Result
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
    
Ranking.to_csv('ranking.csv')
print(Ranking)
