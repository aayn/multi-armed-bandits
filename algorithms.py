import numpy as np
from utils import softmax

class BaseAlgorithm:
    def __init__(self, n_actions):
        self._n_actions = n_actions
        # Action-value estimates
        self._Q = np.array([0.0 for _ in range(n_actions)])
        self._alg_name = 'base'
    
    @property
    def alg_name(self):
        return self._alg_name
    
    def reset(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def parameters(self):
        raise NotImplementedError
    
    def act(self):
        raise NotImplementedError

class EpsilonGreedy(BaseAlgorithm):
    def __init__(self, n_actions, eps=0.0):
        super().__init__(n_actions)
        self._eps = eps
        self._alg_name = 'epsilon_greedy'    
    
    def act(self, step):
        "Choose action for the next time step."
        action = None
        if np.random.random() < self._eps:
            action = np.random.choice(range(self._n_actions))
        else:
            action = np.argmax(self._Q)
        return action


class SampleAverage(EpsilonGreedy):
    def __init__(self, n_actions, eps=0.0):
        super().__init__(n_actions, eps)
        # Reward history
        self._r_hist = np.array([])
        # Action history        
        self._a_hist = np.array([], dtype=np.int)
        self._alg_name = 'sample_average'
    
    def reset(self):
        self._Q = np.array([0.0 for _ in range(self._n_actions)])
        self._r_hist = np.array([])
        self._a_hist = np.array([], dtype=np.int)
    
    def update(self, action, reward, step):
        "Update action-value estimates using sample-averaging."
        for a in range(self._n_actions):
            predicate = (self._a_hist == a)
            psum = np.sum(predicate)
            if psum > 0:
                self._Q[a] = np.sum(self._r_hist[predicate]) / psum

        self._r_hist = np.append(self._r_hist, reward)
        self._a_hist = np.append(self._a_hist, action)

    def parameters(self):
        return {'eps': self._eps}


class OnlineSampleAverage(EpsilonGreedy):
    def __init__(self, n_actions, eps=0.0):
        super().__init__(n_actions, eps)
        # No. of times an action has been taken
        self._N = np.array([0 for _ in range(n_actions)])
        self._alg_name = 'online_sample_average'
    
    def reset(self):
        self._Q = np.array([0.0 for _ in range(self._n_actions)])
        self._N = np.array([0 for _ in range(self._n_actions)])

    def update(self, action, reward, step):
        self._N[action] +=1
        self._Q[action] += (reward - self._Q[action]) / self._N[action]

    def parameters(self):
        return {'eps': self._eps}


class ConstantStepAverage(OnlineSampleAverage):
    def __init__(self, n_actions, eps=0.0, alpha=0.1):
        super().__init__(n_actions, eps)
        self._alpha = alpha
        self._alg_name = 'constant_step_average'
    
    def update(self, action, reward, step):
        self._Q[action] += self._alpha * (reward - self._Q[action])

    def parameters(self):
        return {'eps': self._eps, 'alpha': self._alpha}


class OptimisticInitial(OnlineSampleAverage):
    def __init__(self, n_actions, eps=0.0, Q1=0.0):
        super(n_actions, eps)
        self._Q1 = Q1
        self._Q = np.array([Q1 for _ in range(n_actions)])
        self._alg_name = 'optimistic_initial'
    
    def reset(self):
        self._Q = np.array([self._Q1 for _ in range(self._n_actions)])
        self._N = np.array([0 for _ in range(self._n_actions)])

    def parameters(self):
        return {'eps': self._eps, 'Q1': self._Q1}


class UCB(BaseAlgorithm):
    def __init__(self, n_actions, c):
        super().__init__(n_actions)
        self._c = c
        self._alg_name = 'ucb'
        # No. of times an action has been taken
        self._N = np.array([0 for _ in range(n_actions)])

    def reset(self):
        self._Q = np.array([0.0 for _ in range(self._n_actions)])
        self._N = np.array([0 for _ in range(self._n_actions)])
    
    def update(self, action, reward, step):
        self._N[action] +=1
        self._Q[action] += (reward - self._Q[action]) / self._N[action]
    
    def act(self, step):
        action = np.argmax(self._Q + self._c * np.sqrt(np.log(step) / self._N))
        return action
    
    def parameters(self):
        return {'c': self._c}


class GradientBandit(BaseAlgorithm):
    def __init__(self, n_actions, alpha=0.1):
        super().__init__(n_actions)
        self._alpha = alpha
        self._alg_name = 'gradient_bandit'
        # Action preference
        self._H = np.array([0.0 for _ in range(self._n_actions)])
        # Baseline reward
        self._R = 0.0

    def _pi(self, a):
        return softmax(self._H)[a]
    
    def update(self, action, reward, step):
        H_next = self._H
        for a in range(self._n_actions):
            H_next[a] -=   self._alpha * (reward - self._R) * (1 - self._pi(a))
        H_next[action] +=   self._alpha * (reward - self._R) * (1 - self._pi(action))  
        self._H = H_next
        self._R = (1 - self._alpha) * self._R + self._alpha * reward
    
    def act(self, step):
        return np.random.choice(range(self._n_actions, p=self._pi(slice(None))))
    
    def parameters(self):
        return {'alpha': self._alpha}
