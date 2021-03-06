import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')

Function_list = [rc.elu, rc.tanh, rc.soft_plus, rc.prelu, rc.relu, rc.sigmoid]
# Function_list = [rc.elu, rc.soft_plus, rc.prelu, rc.relu]
for Function_activation in Function_list:
    Path = f'Result/Activation/Sprott/{Function_activation.__name__}'
    Result = pd.read_csv(Path + f'/result_{Function_activation.__name__}.csv', index_col=0)
    print((1 - sum(np.sum(np.isnan(Result))) / (Result.shape[0] * Result.shape[1])) * 100)
    Rou_list = np.unique(Result.index)
    # plt.title(f'{Function_activation.__name__}')
    Result_median = pd.DataFrame(np.zeros((len(Rou_list), Result.shape[1])), 
                                columns=Result.columns, 
                                index=[str(i)[:4] for i in Rou_list])
    for indicator in Result.columns:
        if Function_activation == rc.elu:
            label = 'ELU'
        if Function_activation == rc.tanh:
            label = 'tanh'
        if Function_activation == rc.prelu:
            label = 'PReLU'
        if Function_activation == rc.relu:
            label = 'ReLU'
        if Function_activation == rc.sigmoid:
            label = 'Sigmoid'
        if Function_activation == rc.soft_plus:
            label = 'Softplus'
            
        plt.figure(indicator, figsize=(10, 5))
        for Rou in Rou_list:
            median = Result.loc[Rou][indicator]
            median = np.median(median[~np.isnan(median)])
            Result_median.loc[str(Rou)[:4]][indicator] = median
        # print(Result_median.index[np.argmin(Result_median[indicator])])
        plt.plot(Result_median[indicator], label=label)
        plt.xlabel('$\\rho$')
        plt.ylabel(indicator.upper())
        plt.legend()
        plt.savefig(indicator.upper() + '.svg', format='svg')
    print(np.min(Result_median))
    