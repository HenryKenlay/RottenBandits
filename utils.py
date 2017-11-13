from tqdm import tqdm
import numpy as np
import pandas as pd

def run_experiment(bandit, Agent, experiment_name, T, experiment_repeats):
    print('Running experiment {}'.format(experiment_name), flush=True)
    dfs = []
    bandit.reset()
    opt = np.cumsum(bandit_optimal_policy(bandit, T)[1])
    for i in tqdm(range(experiment_repeats), desc = 'Repeat# '):
        bandit.reset()
        agent = Agent(bandit, T)
        agent.run_agent() 
        regret = opt - np.cumsum(agent.rewards)
        regret = np.concatenate([[0], regret])
        algo = [experiment_name for _ in range(T+1)]
        t = list(range(T+1))
        repeats = [i for _ in range(T+1)]
        dfs.append(pd.DataFrame({'Algo' : algo, 'Repeat' : repeats, 't' : t, 'Regret' : regret}))
    data = pd.concat(dfs)
    data.to_csv('results/{}.csv.gzip'.format(experiment_name), compression = 'gzip')
    return data

def compress_csv(experiment_name):
    data = pd.read_csv('results/{}.csv'.format(experiment_name), index_col = 0)
    data.to_csv('results/{}.csv.gzip'.format(experiment_name), compression = 'gzip')

def bandit_optimal_policy(bandit, T):
    bandit.reset()
    policy = []
    mean_rewards = []
    for i in range(T):
        best_arm, best_reward = 0, 0
        for arm in range(bandit.K):
            if bandit.get_mean(arm) > best_reward:
                best_arm, best_reward = arm, bandit.get_mean(arm)
        policy.append(best_arm)
        mean_rewards.append(best_reward)
        bandit.number_pulls[best_arm] += 1
    return policy, mean_rewards