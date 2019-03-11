import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from matplotlib import rc
import numpy as np
from collections import defaultdict
import pickle
import yaml as y



def custom_violinplot(bandit):
    """Plots the reward distribution for a given bandit.

    bandit: An instance of the Bandit class.
    """
    n_samples = 10000
    dd = defaultdict(lambda: [])
    for _ in range(n_samples):
        for i in range(bandit.n_arms):
            dd[i] += [bandit(i)]
    data = [np.array(dd[i]) for i in range(bandit.n_arms)]
    pos = range(1, bandit.n_arms + 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.violinplot(data, pos, points=100, widths=0.7, showmeans=True,
                   bw_method=0.5)

    for i in range(bandit.n_arms):
        ax.text(i + 1, np.mean(data[i]) + 0.25, rf'$q_*({i+1})$')

    rc('text', usetex=True)

    ax.xaxis.grid(linestyle='dashed', color='black')
    ax.yaxis.grid(linestyle='dashed', color='black')
    plt.xticks(np.arange(11), np.arange(0, 11))
    plt.xlabel('Action/Arm', fontsize=15)
    plt.ylabel('Reward Distribution', fontsize=15)
    plt.show()
    plt.close()


def average_reward(avg_rewards, labels, savepath=None):
    """Plot average reward vs. time steps for algorithms.
    
    avg_rewards: List of average rewards per step (numpy array).
    labels: List of strings for labelling plots.
    savepath: A string; save location for the graph, if any.
    """
    plt.figure(figsize=(20, 20))

    for ar, label in zip(avg_rewards, labels):
        plt.plot(ar, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

    plt.legend()
    
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

def optim_action(oa_percents, labels, savepath=None):
    """Plot % optimal action vs. time steps for algorithms.

    oa_percents: List of percent of times optimal action is taken per
    step (numpy array).
    labels: List of strings for labelling plots.
    savepath: A string; save location for the graph, if any.
    """
    plt.figure(figsize=(20, 20))

    for oap, label in zip(oa_percents, labels):
        plt.plot(oap, label=label)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()


def parameter_study(all_vals, all_pvals, labels, savepath=None):
    """Plot average reward vs. parameter value at which that reward
    is received.

    all_vals: List of list of average rewards. Each inner list contains
    the average reward for all the desired parameter values for a given
    algorithm. Each inner list corresponds to a different algorithm.
    all_pvals: List of list of parameter values. These are the
    corresponding parameter values at which the average rewards are
    obtained.
    labels: List of strings for labelling plots.
    savepath: A string; save location fpr the graph, if any.
    """
    plt.figure(figsize=(20, 20))

    for vals, pvals, label in zip(all_vals, all_pvals, labels):
        plt.plot(pvals, vals, label=label, linewidth=5)
    
    plt.ylabel('Average reward over last 100,000 steps')
    plt.xlabel('Parameter value')
    plt.xscale('log', basex=2)
    xticks = [pow(2, i) for i in range(-7, 3)]
    plt.xticks(xticks)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()