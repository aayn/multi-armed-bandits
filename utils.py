import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from collections import defaultdict


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
