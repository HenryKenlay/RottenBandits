import numpy as np

class Random():
    
    def __init__(self, bandit, T):
        self.bandit = bandit
        self.K = bandit.K
        self.T = T        
        self.policy = []
        self.rewards = []
    
    def run_agent(self):
        for t in range(self.T):
            arm = np.random.choice(range(self.K))
            reward = self.bandit.sample(arm)    
            self.rewards.append(reward)
            self.policy.append(arm)                
        return self.policy, self.rewards
    
class RoundRobin():
    
    def __init__(self, bandit, T):
        self.bandit = bandit
        self.K = bandit.K
        self.T = T        
        self.policy = []
        self.rewards = []
    
    def run_agent(self):
        for t in range(self.T):
            arm = t%self.K
            reward = self.bandit.sample(arm)    
            self.rewards.append(reward)
            self.policy.append(arm)                
        return self.policy, self.rewards