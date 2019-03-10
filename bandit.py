import yaml as y
import numpy as np
from plotting import custom_violinplot

class Bandit:
    def __init__(self, non_stationary=False):
        """Initializes a k-armed bandit.

        Take a look at `config.yml` for different parameters of the
        bandit.
        """
        with open('config.yml') as cfile: 
            config = y.load(cfile)['bandit']

        self._n_arms = config['n_arms']
        self._make_arms(non_stationary)
    
    @property
    def n_arms(self):
        return self._n_arms
    
    def _make_arms(self, non_stationary):
        self._arms = [Arm(non_stationary) for _ in range(self.n_arms)]
    
    def __call__(self, a):
        """Choose one of the actions(arms) of the bandit.

        a: action/arm selected.
        """
        return self._arms[a]()
    
    def plot_reward_dist(self):
        "Plot reward distribution for each action."
        custom_violinplot(self)
    
    def q_star(self, a):
        return self._arms[a].qstar


class Arm:
    def __init__(self, non_stationary=False):
        """Initializes a single bandit arm.

        The values of mean and the variance for the distribution
        of the action corresponding to the arm can be set in
        `config.yml`.
        """
        with open('config.yml') as cfile: 
            config = y.load(cfile)['arm']

        self._mean = config['mean']
        self._var = config['var']
        if non_stationary:
            self._ns_mean = config['ns_mean']
            self._ns_var = config['ns_var']
        else:
            self._ns_mean, self._ns_var = 0.0, 0.0
        
        self._qstar = np.sqrt(self._var) * np.random.randn() + self._mean
    
    def __call__(self):
        self._qstar += np.sqrt(self._ns_var) * np.random.randn() + self._ns_mean
        return np.sqrt(self._var) * np.random.randn() + self._qstar
    
    @property
    def qstar(self):
        return self._qstar


if __name__ == '__main__':
    a = Arm()
    print(a._qstar)
    print(type(a()))
    print(a())
    bandit = Bandit()
    bandit.plot_reward_dist()

        
