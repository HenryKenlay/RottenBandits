from tqdm import tqdm
import numpy as np

class CTO():
    
    def __init__(self, bandit, T):
        self.bandit = bandit
        self.K = bandit.K
        self.T = T        
        self.N = np.zeros(self.K)
        self.policy = []
        self.rewards = []
        self.arm_rewards = [[] for _ in range(self.K)]
        self.theta_hat = np.vectorize(self._theta_hat)
    
    def run_agent(self):
        for t in range(self.T):
            if t < self.K:
                arm = t
            else:
                #detect
                theta_hat = self.theta_hat(range(self.K))
                #balance
                arm = np.argmax((np.floor((self.N+1)/100)+1)**(-theta_hat))
                
            reward = self.bandit.sample(arm)    
            self.N[arm] += 1
            self.arm_rewards[arm].append(reward)
            self.rewards.append(reward)
            self.policy.append(arm)                
            
        return self.policy, self.rewards
    
    def _theta_hat(self, arm):
        best_theta, best_Y = None, np.inf
        sum1 = np.sum(self.arm_rewards[arm])
        for theta in self.bandit.Theta:
            sum2 = np.sum(((np.floor(np.arange(1, self.N[arm]+1)+1)/100)+1)**(-theta))
            Y = np.abs(sum1-sum2)
            if Y < best_Y:
                best_theta, best_Y = theta, Y
        return best_theta
    
class DCTO():
    
    def __init__(self, bandit, T):
        self.bandit = bandit
        self.K = bandit.K
        self.T = T        
        self.N = np.zeros(self.K)
        self.policy = []
        self.rewards = []
        self.arm_rewards = [[] for _ in range(self.K)]
        
    
    def run_agent(self):
        for t in range(self.T):
            if t < self.K:
                arm = t
            else:
                arm = self.get_best_arm(t)
                
            reward = self.bandit.sample(arm)    
            self.N[arm] += 1
            self.arm_rewards[arm].append(reward)
            self.rewards.append(reward)
            self.policy.append(arm)                
            
        return self.policy, self.rewards
    
    def get_best_arm(self, t):
        best_arm, best_arm_reward = None, 0
        for arm in range(self.K):
            arm_reward = self.evaluate_arm(arm, t)
            if arm_reward > best_arm_reward:
                best_arm_reward = arm_reward
                best_arm = arm
        return best_arm
        
    def evaluate_arm(self, arm, t):
        theta_hat = self.theta_hat(arm)
        const_hat = np.mean(self.arm_rewards[arm] - ((np.floor(np.arange(0, self.N[arm])/100)+1)**(-theta_hat)))
        c = np.sqrt((8*np.log(t)*(0.2**2))/self.N[arm])
        return const_hat + (np.floor((self.N[arm]+1)/100)+1)**(-theta_hat) + c
        
    def theta_hat(self, arm):
        Ni = self.N[arm]
        best_theta, best_Y = None, np.inf
        sum1 = sum(self.arm_rewards[arm][:int(Ni/2)]) - sum(self.arm_rewards[arm][int(Ni/2):])
        for theta in self.bandit.Theta:
            Ni = self.N[arm]
            sum2 = np.sum((np.floor(np.arange(0, int(Ni/2))/100)+1)**(-theta)) - np.sum((np.floor(np.arange(int(Ni/2), Ni)/100)+1)**(-theta))
            Y = np.abs(sum1 - sum2)
            if Y < best_Y:
                best_theta, best_Y = theta, Y
        return best_theta
    

        