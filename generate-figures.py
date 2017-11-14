import pandas as pd
import seaborn as sns
import re

#experiment_names = ['P-AV-wSWA', 'P-AV-SWA', 'CTO', 'P-AV-Random']
experiment_names = ['P-ANV-wSWA', 'P-ANV-SWA', 'DCTO', 'P-ANV-Random']

results = []
for experiment in experiment_names:
    results.append(pd.read_csv('results/{}.csv.gzip'.format(experiment), compression = 'gzip'))
data = pd.concat(results)
data.reset_index(inplace=True, drop = True)    
ax = sns.tsplot(data,time = 't', unit = 'Repeat', condition = 'Algo', value = 'Regret', ci = 'sd')
ax.set_ylim(0, 400)

experiments_with_random = [x for x in list(data['Algo'].unique()) if re.search('random', x.lower())]
data_sub = data[~data['Algo'].isin(experiments_with_random)]
ax.set_ylim(0, max(data_sub['Regret']))