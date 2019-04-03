import numpy as np
import numpy.matlib
import cvxpy as cvx
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import block_diag
from joblib import Parallel, delayed
from time import time

from network import Network
from rectangle import Rectangle
from robust_program import RobustProgram
from utils import timing


EPSILON = 0.1
NUM_FEATURES = 2
ACTIVATION = 'relu'
ALPHA = 0
BETA = 1


def main():
    dimension_list = [NUM_FEATURES, 750, 2]
    net = Network(dimension_list, ACTIVATION)

    xc = np.ones((NUM_FEATURES, 1))
    Xin = d2rect(xc)
    Xout = net.forward_prop(Xin)

    rect = Rectangle(xc - EPSILON * np.ones(xc.shape),
                     xc + EPSILON * np.ones(xc.shape))

    # l, u = output_rect(net, rect)
    l, u = parallel_output_rect(net, rect)

    draw_rect(l, u)
    plt.scatter(Xout[0, :], Xout[1, :])
    plt.show()


def d2rect(xc):

    n = 50

    x_min = xc - EPSILON * np.ones((NUM_FEATURES, 1))
    x_max = xc + EPSILON * np.ones((NUM_FEATURES, 1))

    x1 = np.linspace(x_min[0], x_max[0], num=n)
    x2 = np.linspace(x_min[1], x_max[1], num=n)

    X = np.zeros((NUM_FEATURES, n * n))
    X[0, :] = np.matlib.repmat(x1, 1, n).flatten()
    X[1, :] = np.kron(x2, np.ones((1, n))).flatten()

    return X


@timing
def parallel_output_rect(net, rect):
    """Solves SDPs by parallelizing over the number of features"""

    # upper_outputs = Parallel(n_jobs=NUM_FEATURES)(
    #     delayed(parallel_solve_SDP)(k, net, rect, 'upper') for k in range(net.dim_out))
    #
    # lower_outputs = Parallel(n_jobs=NUM_FEATURES)(
    #     delayed(parallel_solve_SDP)(k, net, rect, 'lower') for k in range(net.dim_out))

    types = ['lower', 'upper']
    outputs = Parallel(n_jobs=NUM_FEATURES * len(types))(
        delayed(parallel_solve_SDP)(k, net, rect, b) for b in types for k in range(net.dim_out))

    lower_outputs = outputs[0:NUM_FEATURES]
    upper_outputs = outputs[NUM_FEATURES:]

    return lower_outputs, upper_outputs


def parallel_solve_SDP(idx, net, rect, bound_type):
    """Called by parallel_output_rect to solve SDP for upper/lower bounds"""

    II = np.eye(net.dim_out)
    c = II[:, idx].reshape((net.dim_out, 1))

    if bound_type == 'upper':
        opt = solve_SDP(c, net, rect, verbose=True)

    elif bound_type == 'lower':
        opt = -1. * solve_SDP(-c, net, rect, verbose=True)

    return opt


@timing
def output_rect(net, rect):
    """Find upper and lower bounding lines
    params:
        net: Network instance       - neural network
        rect: Rectangle instance

    returns:
        l, u: lists of ints
        """

    II = np.eye(net.dim_out)

    upper_outputs, lower_outputs = [], []

    for k in range(net.dim_out):
        c = II[:, k].reshape((net.dim_out, 1))

        upper_opt = solve_SDP(c, net, rect, verbose=True)
        lower_opt = -1. * solve_SDP(-c, net, rect, verbose=True)

        upper_outputs.append(upper_opt)
        lower_outputs.append(lower_opt)

    return lower_outputs, upper_outputs


def solve_SDP(c, net, rect, verbose=True):
    """Create and solve SDP to determine support line"""

    sdp = RobustProgram(c, ALPHA, BETA, net, rect)
    solution = sdp.solve()

    if verbose:
        print('Finished solving SDP')
        print('Solution:')
        print(solution)
        print('-' * 50 + '\n')

    return solution


def draw_rect(l, u):
    """Plots bounding rectangle determined by solving SDPs"""

    x1, y1 = l
    x2, y2 = u
    # x1 = l[0]
    # x2 = u[0]
    # y1 = l[1]
    # y2 = u[1]
    plt.plot([x1, x2], [y1, y1])
    plt.plot([x2, x2], [y1, y2])
    plt.plot([x2, x1], [y2, y2])
    plt.plot([x1, x1], [y2, y1])


if __name__ == '__main__':
    main()
