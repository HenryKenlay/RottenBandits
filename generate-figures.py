import pandas as pd
import seaborn as sns

experiment_names = ['P-AV-wSWA', 'P-AV-SWA', 'CTO', 'P-AV-Random']
results = []
for experiment in experiment_names:
    results.append(pd.read_csv('results/{}.csv.gzip'.format(experiment), compression = 'gzip'))
data = pd.concat(results)
data.reset_index(inplace=True, drop = True)    
ax = sns.tsplot(data,time = 't', unit = 'Repeat', condition = 'Algo', value = 'Regret', ci = 'sd')
ax.set_ylim(0, 135)