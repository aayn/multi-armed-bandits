from pathlib import Path
import yaml as y
import numpy as np
import pickle
import multiprocessing as mp
from bandit import Bandit
import algorithms as alg
from storage import Storage


def run(name, algorithm, non_stationary=False, plots=[]):
    """Runs an algorithm with the specified paramters.
    
    name: The plots and other related data of a 'run' is stored under
    data/<name>/.
    algorithm: Instance of an algorithm from `algorithms.py`.
    non_stationary: Whether to run the stationary or non_stationary test bench.
    plots: List of plots to save from `plotting.py`.
    """
    with open('config.yml') as cfile: 
        config = y.load(cfile)['run']

    storage = Storage(name, algorithm)

    for run in range(config['runs']):
        bandit = Bandit(non_stationary)
        print(f'Run number {run + 1}.')

        optim_action = np.argmax([bandit.q_star(a) for a in range(10)])
        optim_action = [] if non_stationary else optim_action
        results = {'rewards': [], 'actions': [], 'optim_action': optim_action}

        for step in range(1, config['steps'] + 1):
            action = algorithm.act(step)
            reward = bandit(action)
            if non_stationary:
                optim_action = np.argmax([bandit.q_star(a) for a in range(10)])
            algorithm.update(action, reward, step)

            results['rewards'].append(reward)
            results['actions'].append(action)
            if non_stationary:
                results['optim_action'].append(optim_action)
        
        storage.update(results['rewards'], results['actions'], results['optim_action'])
        algorithm.reset()
    
    storage.save()


def multi_run(run_tuples):
    """Run multiple algorithms in parallel.
    
    run_tuples: List of tuples, each tuple being the arguments passed
        to one call of `run`.
    """
    procs = []
    for rt in run_tuples:
        proc = mp.Process(target=run, args=rt)
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()


def parameter_search(Algorithm, parameter, start, end, increment_function):
    values = []
    algs = []
    while start <= end:
        values.append(start)
        a = Algorithm(eps=0.1)
        a.edit_parameter(parameter, start)
        a.reset()
        algs.append(a)
        start = increment_function(start)
    optim_procs = mp.cpu_count() - 1
    
    # for i in range(0, len(algs), optim_procs):
    #     run_tuples = list(map(lambda a, v: (f'{a.name}_{v}', a, True), algs[i:i+optim_procs], values[i:i+optim_procs]))
    #     multi_run(run_tuples)
    avg_rewards = list(map(lambda a, v: np.average(Storage(f'{a.name}_{v}').rewards_sum / 1000),
                           algs, values))
    return avg_rewards, values


def plot(name, plots=[]):
    """Draw plots for an already run algorithm.
    
    name: The `name` with which the algorithm previously run.
    plots: List of plots.
    """

if __name__ == '__main__':
    a1 = alg.EpsilonGreedy(eps=0.1)
    # a2 = alg.EpsilonGreedy(eps=0.1, alpha=0.1)
    # a3 = alg.EpsilonGreedy(eps=0.1, Q1=5.0)
    # a4 = alg.UCB(c=2, alpha=0.1)
    # a5 = alg.GradientBandit(alpha=0.1)

    # multi_run([('eps_greedy', a1, True), ('eps_greedy_const', a2, True), ('eps_greedy_optimistic', a3, True),
    #            ('ucb', a4, True), ('gb', a5, True)])
    # ar = parameter_search(alg.EpsilonGreedy, 'eps', 1/128, 1/2, lambda x: x * 2)
    ar = parameter_search(alg.UCB, 'c', 1/16, 4, lambda x: x * 2)
    # print(ar)
