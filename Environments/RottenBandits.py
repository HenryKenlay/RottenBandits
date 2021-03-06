import numpy as np
        
class RottenBandit():
    
    def __init__(self, K):
        self.K = K
        self.reset()
        self.sd = 0.2
    
    def reset(self):
        self.number_pulls = [0 for _ in range(self.K)]

    def sample(self, arm):
        mu = self.get_mean(arm)
        self.number_pulls[arm] += 1
        return np.random.normal(mu, self.sd)

    def get_mean(self, arm):
        print('get mean needs implenenting')

class NonParametricRottenBandit(RottenBandit):
    
    def __init__(self):
        super().__init__(K=2)
        
    def get_mean(self, arm):
        if arm == 0:
            return  0.5
        else:
            if self.number_pulls[1] < 7500:
                return 1
            else:
                return 0.4    

class ParametricRottenBandit(RottenBandit):
    
    def __init__(self, ANV = False, seed = 1):
        super().__init__(K = 10)
        self.seed = seed
        self.ANV = ANV
        self.set_models()
        
    def set_models(self):
        np.random.seed(self.seed)
        self.Theta = np.arange(0.1, 0.4, 0.05) 
        theta = np.random.choice(self.Theta, self.K, replace = True)
        self.theta = theta
        if self.ANV:
            self.additive_c = np.random.uniform(0, 0.5, self.K)
        else:
            self.additive_c = [0 for _ in range(self.K)]
        
    def get_mean(self, arm):
        j = self.number_pulls[arm]
        mu = (np.floor(j/100)+1)**(-self.theta[arm]) + self.additive_c[arm]
        return mu
        