import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from collections import defaultdict
import pickle
from storage import Storage


def custom_violinplot(bandit):
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


def average_reward(storages, savepath=None):
    """Plot average reward vs. time steps given Storage objects of algorithms.
    
    storages: List of Storage objects.
    """
    plt.figure(figsize=(10, 20))
    for st in storages:
        avg_rewards = np.average(st.all_rewards, axis=0)
        params = '; '.join(f'{p} = {v}' for p, v in st.alg_parameters.items())
        label = f'{st.alg_name}; {params}'
        plt.plot(avg_rewards, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    # plt.show()
    if savepath is not None:
        plt.savefig(savepath)
    plt.close()

def optim_action(storages, savepath=None):
    plt.figure(figsize=(10, 20))
    for st in storages:
        runs, steps = st.all_actions.shape
        t = (st.all_actions == st.optim_actions.reshape(runs, -1))
        num_correct = np.sum(t, axis=0)
        percent_correct = 100.0 * (num_correct / runs)
        
        params = '; '.join(f'{p} = {v}' for p, v in st.alg_parameters.items())
        label = f'{st.alg_name}; {params}'
        plt.plot(percent_correct, label=label)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    # plt.show()
    if savepath is not None:
        plt.savefig(savepath)
    plt.close()


if __name__ == '__main__':
    s1 = Storage('eps_greedy')
    s2 = Storage('eps_greedy_const')
    s3 = Storage('eps_greedy_optimistic')
    s4 = Storage('ucb')
    s5 = Storage('gb')
    average_reward([s1, s2, s3, s4, s5])