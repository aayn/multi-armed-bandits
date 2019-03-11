"Reinforcement Learning: An Introduction Chapter 2 Exercises."

import algorithms as alg
from run import multi_run, parameter_search
import plotting as plot
from storage import Storage


def ex_2_5():
    """Design and conduct an experiment to demonstrate the
    difficulties that sample-average methods have for nonstationary problems.
    Use a modified version of the 10-armed testbed in which all the q*(a) start
    out equal and then take independent random walks (say by adding a normally
    distributed increment with mean zero and standard deviation 0.01 to all the
    q*(a) on each step). Prepare plots like Figure 2.2 for an action-value
    method using sample averages, incrementally computed, and another
    action-value method using a constant step-size parameter, alpha = 0.1. Use
    eps = 0.1 and longer runs, say of 10,000 steps."""

    inc = alg.EpsilonGreedy(eps=0.1)
    const = alg.EpsilonGreedy(eps=0.1, alpha=0.1)

    # multi_run([('incremental_2.5', inc, True),
            #    ('const-step_2.5', const, True)])

    s1 = Storage('incremental_2.5')
    s2 = Storage('const-step_2.5')

    plot.average_reward([s1, s2], 'data/exercise_plots/2.5_avgrew.png')
    plot.optim_action([s1, s2], 'data/exercise_plots/2.5_optact.png')


def ex_2_11():
    a1 = alg.EpsilonGreedy(eps=0.1)

    avg_rewards, pvalues = parameter_search(alg.EpsilonGreedy, 'eps', 1/128, 1/2, lambda x: 2 * x)
    # avg_rewards, pvalues = parameter_search(alg.UCB, 'c', 1/16, 4, lambda x: x * 2)    
    plot.parameter_study(avg_rewards, pvalues)


if __name__ == '__main__':
    # ex_2_5()
    ex_2_11()
