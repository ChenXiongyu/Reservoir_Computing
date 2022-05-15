import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import reservoir_computing as rc

import warnings
warnings.filterwarnings('ignore')

Function_list = [rc.elu, rc.tanh, rc.soft_plus, rc.prelu, rc.relu, rc.sigmoid]
for Function_activation in Function_list:
    Path = f'Result/Activation/Roessler/{Function_activation.__name__}'
    Result = pd.read_csv(Path + f'/result_{Function_activation.__name__}.csv', index_col=0)
    Rou_list = np.unique(Result.index)
    plt.figure(figsize=(10, 5))
    plt.title(f'{Function_activation.__name__}')
    Result_median = pd.DataFrame(np.zeros((len(Rou_list), Result.shape[1])), 
                                columns=Result.columns, 
                                index=[str(i)[:4] for i in Rou_list])
    for indicator in Result.columns:
        for Rou in Rou_list:
            median = Result.loc[Rou][indicator]
            median = np.median(median[~np.isnan(median)])
            Result_median.loc[str(Rou)[:4]][indicator] = median
        print(Result_median.index[np.argmin(Result_median[indicator])])
        plt.plot(Result_median[indicator], label=indicator)
        plt.legend()
        