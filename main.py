from pathlib import Path
import yaml as y
import numpy as np
import pickle
import multiprocessing as mp
from bandit import Bandit
import algorithms as al
from storage import Storage


def run(name, algorithm, non_stationary=False, plots=[], config_path=Path('config.yml')):
    """Runs an algorithm with the specified paramters.
    
    name: The plots and other related data of a 'run' is stored under
    data/<name>/.
    algorithm: Instance of an algorithm from `algorithms.py`.
    non_stationary: Whether to run the stationary or non_stationary test bench.
    plots: List of plots to save from `plotting.py`.
    config_path: A Path object to the `config.yml` file.
    """
    with config_path.open() as cfile: 
        config = y.load(cfile)['run']

    storage = Storage(name, algorithm)

    for run in range(config['runs']):
        bandit = Bandit(non_stationary, config_path=config_path)
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


def plot(name, plots=[]):
    """Draw plots for an already run algorithm.
    
    name: The `name` with which the algorithm previously run.
    plots: List of plots.
    """

if __name__ == '__main__':
    a1 = al.OnlineSampleAverage(10, eps=0.1)
    a2 = al.UCB(10, 2)
    a3 = al.GradientBandit(10)

    multi_run([('nonassoc_osa', a1, True), ('nonassoc_ucb', a2, True), ('nonassoc_gb', a3, True)])