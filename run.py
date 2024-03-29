from pathlib import Path
import yaml as y
import numpy as np
import pickle
import multiprocessing as mp
from bandit import Bandit
import algorithms as alg


def run(algorithm, non_stationary=False):
    """Runs an algorithm with the specified paramters.

    algorithm: Instance of an algorithm from `algorithms.py`.
    non_stationary: Whether to run the stationary or non_stationary test bench.
    """
    with open('config.yml') as cfile: 
        config = y.load(cfile)['run']
    runs, steps = config['runs'], config['steps']

    avg_rewards, optim_action_percent = np.zeros(steps), np.zeros(steps)

    for run in range(runs):
        bandit = Bandit(non_stationary)
        print(f'Run number {run + 1}.')

        # One-run rewards
        or_rewards = []
        # One-run actions
        or_actions = []
        optim_action = np.argmax([bandit.q_star(a) for a in range(10)])
        # One-run optimal actions
        or_optim_actions = [] if non_stationary else optim_action

        for step in range(1, steps + 1):
            action = algorithm.act(step)
            reward = bandit(action)
            if non_stationary:
                optim_action = np.argmax([bandit.q_star(a) for a in range(10)])
            algorithm.update(action, reward, step)

            or_rewards.append(reward)
            or_actions.append(action)
            if non_stationary:
                or_optim_actions.append(optim_action)
        
        avg_rewards += or_rewards

        if non_stationary:
            a, o = np.array(or_actions), np.array(or_optim_actions)
        else:
            a, o = np.array(or_actions), or_optim_actions
        optim_action_percent += (a == o)
        
        algorithm.reset()
    
    avg_rewards /= runs
    optim_action_percent = (optim_action_percent / runs) * 100.0

    return avg_rewards, optim_action_percent


def multi_run(*run_tuples):
    """Run multiple algorithms in parallel.
    
    run_tuples: Each tuple is passed to one call of `run`.
    """
    with mp.Pool(processes=(mp.cpu_count() - 1)) as pool:
        results = pool.starmap(run, run_tuples)
    print(results)

    return results


if __name__ == '__main__':
    a1 = alg.EpsilonGreedy(eps=0.1)
    a2 = alg.EpsilonGreedy(eps=0.1, alpha=0.1)
    a3 = alg.EpsilonGreedy(eps=0.1, Q1=5.0)
    a4 = alg.UCB(c=2, alpha=0.1)
    a5 = alg.GradientBandit(alpha=4)

    multi_run((a1, True), (a2, True), (a3, True), (a4, True), (a5, True))