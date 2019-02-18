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


def average_reward(storages):
    """Plot average reward vs. time steps given Storage objects of algorithms.
    
    storages: List of Storage objects.
    """
    plt.figure(figsize=(10, 20))
    for storage in storages:
        avg_rewards = np.average(storage.all_rewards, axis=0)
        plt.plot(avg_rewards, label=f'eps = {storage.eps:.2f}')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()
    plt.close()

def optim_action(storages):
    plt.figure(figsize=(10, 20))
    for st in storages:

        print(st.all_actions[1999,:])
        print(st.optim_actions[1998])

        t = (st.all_actions == st.optim_actions.reshape(2000, -1))
        num_correct = np.sum(t, axis=0)
        print(num_correct)
        print(num_correct.shape)
        percent_correct = 100.0 * (num_correct / 2000.0)

        plt.plot(percent_correct, label=f'eps = {st.eps:.2f}')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.show()
    plt.close()


if __name__ == '__main__':
    # average_reward([Storage('samp_avg_0.05')])
    optim_action([Storage('o_savg_0'), Storage('o_savg_0.01'), Storage('o_savg_0.1')])