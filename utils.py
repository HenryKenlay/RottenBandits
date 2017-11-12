from tqdm import tqdm
import numpy as np
import pandas as pd

def run_experiment(Bandit, Agent, experiment_name, T, experiment_repeats):
    print('Running experiment {}'.format(experiment_name), flush=True)
    dfs = []
    opt = np.cumsum(bandit_optimal_policy(Bandit, T)[1])
    for i in tqdm(range(experiment_repeats), desc = 'Repeat# '):
        bandit = Bandit()
        agent = Agent(bandit, T)
        agent.run_agent() 
        regret = opt - np.cumsum(agent.rewards)
        regret = np.concatenate([[0], regret])
        algo = [experiment_name for _ in range(T+1)]
        t = list(range(T+1))
        repeats = [i for _ in range(T+1)]
        dfs.append(pd.DataFrame({'Algo' : algo, 'Repeat' : repeats, 't' : t, 'Regret' : regret}))
    return pd.concat(dfs)

def bandit_optimal_policy(Bandit, T):
    bandit = Bandit()
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