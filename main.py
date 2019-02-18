from pathlib import Path
import yaml as y
import numpy as np
import pickle
import multiprocessing as mp
from bandit import Bandit
from algorithms import SampleAverage, OnlineSampleAverage
from storage import Storage


def run(name, algorithm, plots=[], config_path=Path('config.yml')):
    """Runs an algorithm with the specified paramters.
    
    name: The plots and other related data of a 'run' is stored under
    data/<name>/.
    Algorithm: Pick an algorithm from `algorithms.py`.
    plots: List of plots to save from `plotting.py`.
    config_path: A Path object to the `config.yml` file.
    """
    with config_path.open() as cfile: 
        config = y.load(cfile)['run']
    
    # all_rewards = []
    # all_actions = []
    # best_actions = []
    storage = Storage(name)

    for run in range(config['runs']):
        bandit = Bandit(config_path=config_path)
        if run == 0:
            storage.update(algorithm.info())
        print(f'Run number {run + 1}.')

        optim_action = np.argmax([bandit.q_star(a) for a in range(10)])
        run_info = {'rewards': [], 'actions': [], 'optim_action': optim_action}

        for step in range(1, config['steps'] + 1):
            action = algorithm.act()
            reward = bandit(action)
            algorithm.update(action, reward, step)

            run_info['rewards'].append(reward)
            run_info['actions'].append(action)
            # print(action, reward)
        
        algorithm.reset()    
        storage.update(run_info)
    
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
    multi_run([('o_savg_0', OnlineSampleAverage(10, eps=0.0)),
               ('o_savg_0.01', OnlineSampleAverage(10, eps=0.01)),
               ('o_savg_0.1', OnlineSampleAverage(10, eps=0.1))])