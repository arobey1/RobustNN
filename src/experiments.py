import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from robust_tools import test_robustness
from network import Network


TESTS = ['RAND']

def main():

    if 'RAND' in TESTS:
        in_dim, out_dim = 2, 2
        hidden_sizes = [5, 10, 15, 20]
        epsilon = 0.1

        results = random_tests(in_dim, out_dim, hidden_sizes, epsilon)
        plot_test_resultls(results, hidden_sizes, plot_type='rand')

    if 'EPSILON' in TESTS:
        in_dim, out_dim = 2, 2
        hidden_dim = 100
        epsilons = [0.01, 0.1, 0.5, 1]

        results = epsilon_tests(in_dim, out_dim, hidden_dim, epsilons)
        plot_test_resultls(results, epsilons, plot_type='epsilon')


def random_tests(in_dim, out_dim, hidden_sizes, epsilon, solver='cvxopt'):
    """Test networks with varying number of hidden layers and random weights

    params:
        in_dim, out_dim: ints       - input and output dimension of NN
        hidden_sizes: list of ints  - list of hidden sizes to test
        epsilon: float              - robustness parameter

    returns:
        results: dict
    """

    results = dict.fromkeys(hidden_sizes)

    for h in hidden_sizes:
        net = Network([in_dim, h, out_dim], 'relu', 'rand')
        lower, upper, out_data = test_robustness(net, epsilon, solver=solver,
                                                    parallel=True)
        net.save_params('weights')
        results[h] = {
            'lower': lower,
            'upper': upper,
            'data': out_data,
        }

    return results

def epsilon_tests(in_dim, out_dim, hid_dim, epsilons):

    results = dict.fromkeys(epsilons)
    net = Network([in_dim, 100, out_dim], 'relu', 'rand', load_weights=True,
                    weight_root='saved_weights')

    for e in epsilons:

        lower, upper, out_data = test_robustness(net, e, parallel=True)
        results[e] = {
            'lower': lower,
            'upper': upper,
            'data': out_data,
        }

    return results


def plot_test_resultls(results, sizes, plot_type):

    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(12,5))
    area = np.pi

    for idx, h in enumerate(sizes):

        x1, y1 = results[h]['lower']
        x2, y2 = results[h]['upper']
        curr_ax = ax[idx]

        if idx == 0:
            curr_ax.scatter(results[h]['data'][0,:], results[h]['data'][1,:], s=area,
                            label='Network output')
            curr_ax.plot([x1, x2], [y1, y1], color='red', label='Outer approxmation')

        else:
            curr_ax.scatter(results[h]['data'][0,:], results[h]['data'][1,:], s=area)
            curr_ax.plot([x1, x2], [y1, y1], color='red')

        curr_ax.plot([x1, x2], [y1, y1], color='red')
        curr_ax.plot([x2, x2], [y1, y2], color='red')
        curr_ax.plot([x2, x1], [y2, y2], color='red')
        curr_ax.plot([x1, x1], [y2, y1], color='red')
        curr_ax.grid()

        if plot_type == 'rand':
            curr_ax.set_title(str(h) + ' hidden units', fontdict={'fontsize': 8})
        elif plot_type == 'epsilon':
            curr_ax.set_title(r'$\epsilon = $' + str(h), fontdict={'fontsize': 8})

    fig.legend(ncol=2, loc='lower center')

    if plot_type == 'rand':
        fig.suptitle('Impact of number of hidden layers on outer approximation')
    elif plot_type == 'epsilon':
        fig.suptitle(r'Impact of $\epsilon$ on outer approximation')

    plt.show()




if __name__ == '__main__':
    main()
