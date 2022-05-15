import numpy as np
import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')


# Trajectory
Function_trajectory = rc.sprott

# Capacity
Capacity_training = 5000
Capacity_predicting = 1500

# Parameters
N = [500, 500]
D = 3
Beta = 1e-4
Sigma = [1, 1]
Rou = [0.1, 0.1]

# Function
Function_activation = [rc.soft_plus, rc.soft_plus]
Function_basis_1 = rc.original
Function_basis_2 = rc.square

# Training Process
Start_pos = list(np.random.rand(int(D)))
Time_training, Trajectory_training = \
    Function_trajectory(length=Capacity_training, sample=0.01, 
                        x0=Start_pos, discard=0)

# n, d, rou, sigma, beta, trajectory_training, plot=True,
#           activation_function=np.tanh, 
#           basis_function_1=original, basis_function_2=np.square, 
#           discard=1000
          
def train_parallel(n, d, rou, sigma, beta, trajectory_training, function_activation, 
                   function_basis_1=rc.original, function_basis_2=rc.original, plot=True):
    
    
    def reservoir_single(n, d, rou, sigma, trajectory_training, activation_function):
        # print('Train Process...')
        w_r_function = rc.reservoir_construction_fix_degree
        w_i_function = rc.reservoir_construction_average_allocate

        w_r = w_r_function(n, n, 'uniform', d, sr=rou,  low=0.0, high=1.0)
        w_i = w_i_function(n, d, 'uniform', low=-sigma, high=sigma)

        reservoir_start = np.zeros(n)
        reservoir_state_training = np.zeros((len(trajectory_training), len(reservoir_start)))
        reservoir_state_training[0, :] = reservoir_start

        for i in range(1, len(trajectory_training)):
            reservoir_state_training[i, :] = activation_function(np.dot(w_r, reservoir_state_training[i - 1, :]) +
                                                                np.dot(w_i, trajectory_training[i - 1, :]))


        return w_r, w_i, reservoir_state_training


    def reservoir_regression(beta, trajectory_training, reservoir_state_training, plot, 
                             basis_function_1, basis_function_2, discard):
        x = reservoir_state_training[discard:, :]
        y = trajectory_training[discard:, :]

        s = x.copy()
        s[:, ::2] = basis_function_1(s[:, ::2])
        s[:, 1::2] = basis_function_2(s[:, 1::2])
        w_0 = np.linalg.solve(np.dot(s.T, s) + beta * np.eye(s.shape[1]), np.dot(s.T, y))
        w_0 = w_0.T

        w_01 = np.zeros(w_0.shape)
        w_02 = np.zeros(w_0.shape)

        w_01[:, ::2] = w_0[:, ::2]
        w_02[:, 1::2] = w_0[:, 1::2]

        output_training = np.dot(w_01, basis_function_1(x.T)) + np.dot(w_02, basis_function_2(x.T))
        output_training = output_training.T

        def f_out(r):
            return (np.dot(w_01, basis_function_1(r.T)) + np.dot(w_02, basis_function_2(r.T))).T

        if plot:
            rc.plot_trajectory(y, output_training)
        
        return f_out, output_training


    reservoir_state_training = []
    
    for r in range(len(n)):
        
        n_r = n[r]
        rou_r = rou[r]
        sigma_r = sigma[r]
        function_activation_r = function_activation[r]
        
        w_r_r, w_i_r, reservoir_state_training_r = \
            reservoir_single(n_r, d, rou_r, sigma_r, trajectory_training,
                             activation_function=function_activation_r)
        reservoir_state_training.append(reservoir_state_training_r)

    reservoir_state_training = np.hstack(reservoir_state_training)

    f_out, output_training = \
        reservoir_regression(beta, trajectory_training, reservoir_state_training, plot, 
                             basis_function_1=rc.original, basis_function_2=np.square, discard=1000)





def predict_wide(w_r_list, w_i_list, f_out, 
                 trajectory_predicting, reservoir_state_predicting,
                 function_activation_list, plot=True, save_path=''):
    # print('Self Predicting Process...')
    output_predicting = np.zeros(trajectory_predicting.shape)
    output_predicting[0, :] = trajectory_predicting[0, :]

    for i in range(1, len(trajectory_predicting)):
        for r in range(reservoir_num):
        reservoir_state_predicting[i, :] = activation_function(np.dot(w_r, reservoir_state_predicting[i - 1, :]) +
                                                               np.dot(w_i, output_predicting[i - 1, :]))
        output_predicting[i, :] = f_out(reservoir_state_predicting[i, :])

    if plot:
        plot_trajectory(trajectory_predicting, output_predicting)
        
        if save_path:
            plt.savefig(save_path, format='svg')
            plt.close()

    return output_predicting

# Predicting Process
time_predicting, trajectory_predicting = \
    rc.roessler(length=1000, sample=0.01, discard=0, 
                x0=list(trajectory_training[-1, :]))

reservoir_state_predicting = np.zeros((1000, sum(n_list)))
reservoir_state_predicting[0, :] = reservoir_state_training[-1, :]

output_predicting = \
    rc.predict(W_r, W_i, F_out, Trajectory_predicting, 
               Reservoir_state_predicting,
               activation_function=Function_activation, 
               plot=True)















# # Trajectory
# Function_trajectory = rc.roessler

# # Capacity
# Capacity_training = 5000
# Capacity_predicting = 1500


# D = 3
# Beta = 1e-4
# Sigma = 1
# Rou = 0.1

# # Function
# Function_basis_1 = rc.original
# Function_basis_2 = rc.square

# # Training Process
# Start_pos = list(np.random.rand(int(D)))
# Time_training, Trajectory_training = \
#     Function_trajectory(length=Capacity_training, sample=0.01, 
#                         x0=Start_pos, discard=0)




# W_r, W_i, F_out, Reservoir_state_training, Output_training = \
#     rc.train(N, D, Rou, Sigma, Beta, Trajectory_training, plot=True,
#              basis_function_1=Function_basis_1, 
#              basis_function_2=Function_basis_2,
#              activation_function=Function_activation)





# # Predicting Process
# Time_predicting, Trajectory_predicting = \
#     Function_trajectory(length=Capacity_predicting, 
#                         sample=0.01, discard=0, 
#                         x0=list(Trajectory_training[-1, :]))

# Reservoir_state_predicting = np.zeros((Capacity_predicting, N))
# Reservoir_state_predicting[0, :] = Reservoir_state_training[-1, :]

# Output_predicting = \
#     rc.predict(W_r, W_i, F_out, Trajectory_predicting, 
#                Reservoir_state_predicting,
#                activation_function=Function_activation, 
#                plot=True)

# # Valuation
# Distance, Evaluation = rc.error_evaluate(Trajectory_predicting, 
#                                          Output_predicting,
#                                          Time_predicting, plot=True)
