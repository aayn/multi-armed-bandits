import yaml as y
from pathlib import Path
import numpy as np
from utils import custom_violinplot

class Bandit:
    def __init__(self, config_path=Path('config.yml')):
        """Initializes a k-armed bandit.

        Take a look at `config.yml` for different parameters of the
        bandit.
        """
        config = y.load(config_path.open())['bandit']
        self._n_arms = config['n_arms']
        self._make_arms()
    
    @property
    def n_arms(self):
        return self._n_arms
    
    def _make_arms(self):
        self._arms = [Arm() for _ in range(self.n_arms)]
    
    def __call__(self, a):
        """Choose one of the actions(arms) of the bandit.

        a: action/arm selected.
        """
        return self._arms[a]()
    
    def plot_reward_dist(self):
        "Plot reward distribution for each action."
        custom_violinplot(self)


class Arm:
    def __init__(self, config_path=Path('config.yml')):
        """Initializes a single bandit arm.

        The values of mean and the variance for the distribution
        of the action corresponding to the arm can be set in
        `config.yml`.
        """
        config = y.load(config_path.open())['arm']
        self._mean = config['mean']
        self._var = config['var']

        
        self._qstar = np.sqrt(self._var) * np.random.randn() + self._mean
    
    def __call__(self):
        return np.sqrt(self._var) * np.random.randn() + self._qstar


if __name__ == '__main__':
    a = Arm()
    print(a._qstar)
    print(type(a()))
    print(a())
    bandit = Bandit()
    bandit.plot_reward_dist()

        
