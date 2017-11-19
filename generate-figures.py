import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
plt.clf()
experiment_names = ['NP-Random', 'NP-wSWA', 'NP-SWA']
results = []
for experiment in experiment_names:
    results.append(pd.read_csv('results/{}.csv.gzip'.format(experiment), compression = 'gzip'))
data = pd.concat(results)
data.reset_index(inplace=True, drop = True)    
ax = sns.tsplot(data,time = 't', unit = 'Repeat', condition = 'Algo', value = 'Regret', ci = 'sd')
ax.set_xlabel("Time steps")
legend = ax.legend(loc = 2)
legend.set_label('')
legend.get_texts()[0].set_text('Random')
legend.get_texts()[1].set_text('wSWA')
legend.get_texts()[2].set_text('SWA')
ax.set_title('Non-Parametric Case')
ax.grid()
ax.get_figure().savefig('figures/non-parametric.pdf', bbox_inches = 'tight')

#%%
plt.clf()
experiment_names = ['P-AV-Random','P-AV-wSWA', 'P-AV-SWA', 'CTO', 'AV-DCTO']
results = []
for experiment in experiment_names:
    results.append(pd.read_csv('results/{}.csv.gzip'.format(experiment), compression = 'gzip'))
data = pd.concat(results)
data.reset_index(inplace=True, drop = True)    
ax = sns.tsplot(data,time = 't', unit = 'Repeat', condition = 'Algo', value = 'Regret', ci = 'sd')
ax.set_xlabel("Time steps")
legend = ax.legend(loc = 2)
legend.set_label('')
legend.get_texts()[0].set_text('Random')
legend.get_texts()[1].set_text('wSWA')
legend.get_texts()[2].set_text('SWA')
legend.get_texts()[3].set_text('CTO')
legend.get_texts()[4].set_text('DCTO')
ax.set_title('Asymptotically Vanishing Case')
ax.set_ylim(0, 130)
ax.grid()
ax.get_figure().savefig('figures/AV.pdf', bbox_inches = 'tight')


#%%
plt.clf()
experiment_names = ['P-ANV-Random', 'P-ANV-wSWA', 'P-ANV-SWA', 'DCTO']
results = []
for experiment in experiment_names:
    results.append(pd.read_csv('results/{}.csv.gzip'.format(experiment), compression = 'gzip'))
data = pd.concat(results)
data.reset_index(inplace=True, drop = True)    
ax = sns.tsplot(data,time = 't', unit = 'Repeat', condition = 'Algo', value = 'Regret', ci = 'sd')
ax.set_xlabel("Time steps")
legend = ax.legend(loc = 2)
legend.set_label('')
legend.get_texts()[0].set_text('Random')
legend.get_texts()[1].set_text('wSWA')
legend.get_texts()[2].set_text('SWA')
legend.get_texts()[3].set_text('DCTO')
ax.set_title('Asymptotically Non-Vanishing Case')
ax.set_ylim(0, 420)
ax.grid()
ax.get_figure().savefig('figures/ANV.pdf', bbox_inches = 'tight')