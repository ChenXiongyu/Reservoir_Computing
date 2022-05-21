import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

import warnings
warnings.filterwarnings('ignore')

Root = 'Result/Kuramoto/'
Path_list = os.listdir(Root)
for P in Path_list:
    Result = pd.read_csv(Root + P + f'/result_{P}.csv', index_col=0)
    print((1 - sum(np.sum(np.isnan(Result))) / (Result.shape[0] * Result.shape[1])) * 100)
    Rou_list = np.unique(Result.index)
    # plt.title(f'{Function_activation.__name__}')
    Result_median = pd.DataFrame(np.zeros((len(Rou_list), Result.shape[1])), 
                                 columns=Result.columns, 
                                 index=[str(i)[:4] for i in Rou_list])
    for indicator in Result.columns:
        plt.figure(indicator, figsize=(10, 5))
        for Rou in Rou_list:
            median = Result.loc[Rou][indicator]
            median = np.median(median[~np.isnan(median)])
            Result_median.loc[str(Rou)[:4]][indicator] = median
        plt.plot(Result_median[indicator], label=int(np.where(np.array(Path_list) == P)[0]) + 1)
        plt.xlabel('$\\rho$')
        plt.ylabel(indicator.upper())
        plt.legend()
        plt.savefig(indicator.upper() + '.svg', format='svg')
    print(np.min(Result_median))
        