import pickle
import yaml as y
import numpy as np
from pathlib import Path


class Storage:
    "Standard storage format used for this project."

    def __init__(self, name, algorithm=None):
        with open('config.yml') as cfile: 
            run_config = y.load(cfile)['run']
        runs, steps = run_config['runs'], run_config['steps']

        self._name = name
        self._path = Path(f'data/{name}/')
        if algorithm is None:
            self.load()
        else:
            if not self._path.exists():
                self._path.mkdir()

            self._rewards_sum = np.zeros(steps)
            self._optim_action_count = np.zeros(steps)
            self._alg_name = algorithm.name
            self._alg_params = algorithm.parameters
    
    @property
    def rewards_sum(self):
        return self._rewards_sum
    
    @property
    def optim_action_count(self):
        return self._optim_action_count
    
    @property
    def path(self):
        return self._path
    
    @property
    def alg_parameters(self):
        return self._alg_params
    
    @property
    def alg_name(self):
        return self._alg_name
    
    def update(self, rewards, actions, optim_actions):
        """Updates the stored arrays.
        
        rewards: An array-like object of all the rewards of one run.
        action: An array-like object of all the actions of one run.
        optim_actions: An array-like object or a single number. In the
            non-stationary case, where the optimal action can change
            every step, pass an array. In the stationary case, a single
            number is valid too.
        """
        self._rewards_sum += rewards
        if isinstance(optim_actions, int):
            a, o = np.array(actions), optim_actions
        else:
            a, o = np.array(actions), np.array(optim_actions)
        self._optim_action_count += (a == o)
    
    def save(self):
        with open(f'data/{self._name}/storage.pkl', 'wb') as sfile:
            pickle.dump([self._rewards_sum, self._optim_action_count,
                         self._alg_name, self._alg_params], sfile)
    
    def load(self):
        with open(f'data/{self._name}/storage.pkl', 'rb') as sfile:
            self._rewards_sum, self._optim_action_count, self._alg_name, self._alg_params = pickle.load(sfile)

