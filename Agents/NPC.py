import numpy as np

class SWA():
    
    def __init__(self, bandit, T, alpha = 2**(-2/3)):
        self.bandit = bandit
        self.K = bandit.K
        self.T = T
        self.alpha = alpha
        self.N = [0 for _ in range(self.K)]
        self.M = int(self.calculate_M())
        self.policy = []
        self.rewards = []
        self.arm_rewards = [[] for _ in range(self.K)]
        
    def calculate_M(self, sigma = 0.2):
        M = []
        M.append(self.alpha)
        M.append(4**(2/3))
        M.append(sigma**(2/3))
        M.append(self.K**(-2/3))
        M.append(self.T**(2/3))
        M.append(np.log(np.sqrt(2)*self.T)**(1/3))
        M = np.prod(M)
        return np.ceil(M)
    
    def run_agent(self):
        for t in range(self.T):
            if t < self.K*self.M:
                # explore
                arm = t%self.K
                reward = self.bandit.sample(arm)
            else:
                best_i, best_sum = 0, 0
                for i in range(self.K):
                    lower = self.N[i] - self.M + 1
                    upper = self.N[i]
                    loopsum = sum(self.arm_rewards[i][lower:upper+1])/self.M
                    if loopsum > best_sum:
                        best_sum, best_i = loopsum, i
                arm = best_i
                reward = self.bandit.sample(arm)
                
            self.N[arm] += 1
            self.arm_rewards[arm].append(reward)
            self.rewards.append(reward)
            self.policy.append(arm)
        return self.policy, self.rewards

class wSWA():
    
    def __init__(self, bandit, unknown_T, alpha = 2**(-2/3)):
        self.bandit = bandit
        self.unknown_T = unknown_T
        self.alpha = alpha
        self.rewards = []
        self.policy = []
        
    def run_agent(self):
        Ts = [2**i for i in range(int(np.floor(np.log2(self.unknown_T+1))))]
        if self.unknown_T - sum(Ts) > 0:
            Ts = Ts + [self.unknown_T - sum(Ts)]
        for T in Ts:
            sub_instance = SWA(self.bandit, T, self.alpha)
            policy, rewards = sub_instance.run_agent()
            self.policy = self.policy + policy
            self.rewards = self.rewards + rewards
        return self.policy, self.rewards