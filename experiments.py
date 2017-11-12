from Agents.NPC import SWA, wSWA
from RottenBandits import NonParametricRottenBandit, ParametricRottenBandit
from utils import run_experiment
import numpy as np
import pandas as pd
import seaborn as sns


#%% constants
np.random.seed(1)
T = 30000
experiment_repeats = 100



#%% generate results
results = []
experiments = [(ParametricRottenBandit, SWA, 'SWA'),
               (ParametricRottenBandit, wSWA, 'wSWA')]
for experiment in experiments:
    results.append(run_experiment(*experiment, T = T, experiment_repeats = experiment_repeats))
data = pd.concat(results)
data.reset_index(inplace=True, drop = True)

#%% plot results
print('Plotting data')
ax = sns.tsplot(data,time = 't', unit = 'Repeat', condition = 'Algo', value = 'Regret', ci = 'sd')
