import pickle
import numpy as np
from pathlib import Path


class Storage:
    "Standard storage format used for this project."

    def __init__(self, name, algorithm=None):
        self._name = name
        self._path = Path(f'data/{name}/')
        if algorithm is None:
            self.load()
        else:
            try:
                self._path.mkdir()
            except FileExistsError:
                pass
            self._all_rewards = []
            self._all_actions = []
            self._optim_actions = []
            self._alg_name = algorithm.name
            self._alg_params = algorithm.parameters
    
    @property
    def all_rewards(self):
        return np.array(self._all_rewards)
    
    @property
    def all_actions(self):
        return np.array(self._all_actions)
    
    @property
    def optim_actions(self):
        return np.array(self._optim_actions)
    
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
        self._all_rewards.append(rewards)
        self._all_actions.append(actions)
        self._optim_actions.append(optim_actions)
    
    def save(self):
        with open(f'data/{self._name}/storage.pkl', 'wb') as sfile:
            pickle.dump([self._all_rewards, self._all_actions,
                         self._optim_actions, self._alg_name,
                         self._alg_params], sfile)
    
    def load(self):
        with open(f'data/{self._name}/storage.pkl', 'rb') as sfile:
            self._all_rewards, self._all_actions, self._optim_actions, self._alg_name, self._alg_params = pickle.load(sfile)

