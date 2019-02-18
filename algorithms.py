from abc import ABC, abstractmethod
import numpy as np

class BaseAlgorithm(ABC):
    @abstractmethod
    def update(self, action, reward, step):
        ...

    @abstractmethod
    def act(self):
        ...
    
    @abstractmethod
    def reset(self):
        ...

class SampleAverage(BaseAlgorithm):
    def __init__(self, n_actions, eps=0.0, Q1=0.0):
        # Reward history
        self._r_hist = np.array([])
        # Action history        
        self._a_hist = np.array([], dtype=np.int)
        self._n_actions = n_actions
        self._eps = eps
        # Action-value estimates
        self._Q1 = Q1
        self._Q = np.array([Q1 for _ in range(n_actions)])
    
    def reset(self):
        self._Q = np.array([self._Q1 for _ in range(self._n_actions)])
        self._r_hist = np.array([])
        self._a_hist = np.array([], dtype=np.int)

    def info(self):
        return {'eps': self._eps, 'n_actions': self._n_actions}
    
    def update(self, action, reward, step):
        "Update action-value estimates using sample-averaging."
        for a in range(self._n_actions):
            predicate = (self._a_hist == a)
            psum = np.sum(predicate)
            if psum > 0:
                self._Q[a] = np.sum(self._r_hist[predicate]) / psum

        self._r_hist = np.append(self._r_hist, reward)
        self._a_hist = np.append(self._a_hist, action)
        
    
    def act(self):
        "Choose action for the next time step."
        action = None
        if np.random.random() < self._eps:
            action = np.random.choice(range(self._n_actions))
        else:
            action = np.argmax(self._Q)
        return action

    
class OnlineSampleAverage(BaseAlgorithm):
    def __init__(self, n_actions, eps=0.0, Q1=0.0):
        self._n_actions = n_actions
        self._eps = eps
        self._Q1 = Q1
        # Action-value estimates
        self._Q = np.array([Q1 for _ in range(n_actions)])
        # No. of times an action has been taken
        self._N = np.array([0 for _ in range(n_actions)])
    
    def reset(self):
        self._Q = np.array([self._Q1 for _ in range(self._n_actions)])
        self._N = np.array([0 for _ in range(self._n_actions)])

    def info(self):
        return {'eps': self._eps, 'n_actions': self._n_actions}
    
    def update(self, action, reward, step):
        self._N[action] +=1
        self._Q[action] += (reward - self._Q[action]) / self._N[action]

    def act(self):
        "Choose action for the next time step."
        action = None
        if np.random.random() < self._eps:
            action = np.random.choice(range(self._n_actions))
        else:
            action = np.argmax(self._Q)
        return action

    
