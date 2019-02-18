import pickle
import numpy as np
from pathlib import Path


class Storage:
    "Standard storage format used for this project."

    def __init__(self, name):
        self._name = name
        self._path = Path(f'data/{name}/')
        try:
            self.load()
        except FileNotFoundError:
            self._path.mkdir()
            self._all_rewards = []
            self._all_actions = []
            self._optim_actions = []
            self._eps = None
            self._n_actions = 0
    
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
    def eps(self):
        return self._eps
    
    @property
    def n_actions(self):
        return self._n_actions
    
    @property
    def path(self):
        return self._path
    
    def update(self, info):
        if 'rewards' in info:
            self._all_rewards.append(info['rewards'])
            self._all_actions.append(info['actions'])
            self._optim_actions.append(info['optim_action'])
        elif 'eps' in info:
            self._eps = info['eps']
            self._n_actions = info['n_actions']
    
    def save(self):
        with open(f'data/{self._name}/storage.pkl', 'wb') as sfile:
            pickle.dump([self._all_rewards, self._all_actions, self._optim_actions, self._eps,
                         self._n_actions], sfile)
    
    def load(self):
        with open(f'data/{self._name}/storage.pkl', 'rb') as sfile:
            self._all_rewards, self._all_actions, self._optim_actions, self._eps, self._n_actions = pickle.load(sfile)

