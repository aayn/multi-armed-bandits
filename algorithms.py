from collections import OrderedDict
import numpy as np
from utils import softmax


class ActionValueEstimator:
    def __init__(self, n_actions=10, eps=None, alpha=None, Q1=None):
        """Action-value estimation using the epsilon-greedy method.

        n_actions: Number of actions that can be taken.
        eps: Epsilon paramter for epsilon-greedy method.
        alpha: constant per-time-step parameter; if 'None' then sample
            averaging is done.
        Q1: Initial Q-value estimate
        """
        self._n_actions = n_actions
        self._eps = eps
        self._alpha = alpha
        self._Q1 = Q1
        self._name = 'estimate'
        
        self._parameters = OrderedDict()

        if eps is not None:
            self._name += '_eps_greedy'
            self._parameters['eps'] = self._eps
        if alpha is not None:
            self._name += '_const'
            self._parameters['alpha'] = self._alpha
        if Q1 is not None:
            self._name += '_q1'
            self._parameters['Q1'] = self._Q1
        else:
            self._Q1 = 0.0
        self.reset()

    def reset(self):
        # Action-value estimates
        self._Q = np.array([self._Q1 for _ in range(self._n_actions)])
        # Number of times an action is taken
        self._N = np.array([0 for _ in range(self._n_actions)])
    
    def update(self, action, reward, step):
        self._N[action] +=1
        if self._alpha is None:
            self._Q[action] += (reward - self._Q[action]) / self._N[action]
        else:
            self._Q[action] += self._alpha * (reward - self._Q[action])
    
    def act(self, step):
        raise NotImplementedError
    
    @property
    def parameters(self):
        return self._parameters
    
    @property
    def name(self):
        return self._name

class EpsilonGreedy(ActionValueEstimator):
    def __init__(self, n_actions=10, eps=None, alpha=None, Q1=None):
        super().__init__(n_actions, eps, alpha, Q1)

    def act(self, step):
        "Choose action for the next time step."
        action = None
        if np.random.random() < self._eps:
            action = np.random.choice(range(self._n_actions))
        else:
            action = np.argmax(self._Q)
        return action


class UCB(ActionValueEstimator):
    def __init__(self, c, n_actions=10, alpha=None, Q1=None):
        super().__init__(n_actions, None, alpha, Q1)
        self._c = c
        self._parameters['c'] = c
        self._name = 'ucb'
    
    def act(self, step):
        action = np.argmax(self._Q + self._c * np.sqrt(np.log(step) / (self._N + 1e-6)))
        return action


class GradientBandit:
    def __init__(self, n_actions=10, alpha=0.1):
        self._n_actions = n_actions
        self._alpha = alpha
        self._name = 'gradient_bandit'
        self.reset()

    def _pi(self):
        return softmax(self._H)
    
    def reset(self):
        # Action preference
        self._H = np.array([0.0 for _ in range(self._n_actions)])
        # Baseline reward
        self._R = 0.0
        
    def update(self, action, reward, step):
        H_next = self._H
        # Looping over actions is too slow; this is faster
        H_next -=   self._alpha * (reward - self._R) * (1 - self._pi())
        H_next[action] +=   2 * self._alpha * (reward - self._R) * (1 - self._pi()[action])  
        
        self._H = H_next
        self._R = (1 - self._alpha) * self._R + self._alpha * reward
    
    def act(self, step):
        return np.random.choice(range(self._n_actions), p=self._pi())
    
    @property
    def parameters(self):
        return {'alpha': self._alpha}
    
    @property
    def name(self):
        return self._name