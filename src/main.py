
from utils import timing
from robust_program import RobustProgram
from rectangle import Rectangle
from network import Network
from time import time
from joblib import Parallel, delayed
from scipy.linalg import block_diag
import numpy as np
import numpy.matlib
import cvxpy as cvx

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


DATA_PTS_PER_FEATURE = 50
EPSILON = 0.1
NUM_FEATURES = 2
NUM_CLASSES = 2
NUM_HIDDEN = 50
ACTIVATION = 'relu'
ALPHA = 0
BETA = 1


def main():

    # point at which we test the robustness of the neural network
    x_center = np.ones((NUM_FEATURES, 1))

    # generated data in the rectangle defined by EPSILON & x_center
    input_data, x_lower_bound, x_upper_bound = create_rect_data(x_center)
    rect = Rectangle(x_lower_bound, x_upper_bound)

    # create neural network model
    dimension_list = [NUM_FEATURES, NUM_HIDDEN, NUM_CLASSES]
    net = Network(dimension_list, ACTIVATION, rect, 'rand')

    # pass the input data through the network (as if the network were already trained)
    network_output = net.forward_prop(input_data)

    l, u = output_rect(net, rect)
    # l, u = parallel_output_rect(net, rect)

    draw_rect(l, u)
    plt.scatter(network_output[0, :], network_output[1, :])
    plt.show()


def create_rect_data(x_center):
    """Create matrix of input data for neural network

    params:
        x_center - (NUM_FEATURES, 1) vector - point at which we test robustness

    returns:
        X: (NUM_FEATURES, n_points ** NUM_FEATURES) matrix  - input data
        x_min: (NUM_FEATURES, 1) vector                     - min possible input vector
        x_max: (NUM_FEATURES, 1) vector                     - max possible input vector
    """

    # find maximum and minimum possible deviations from x_center
    x_min = x_center - EPSILON * np.ones((NUM_FEATURES, 1))
    x_max = x_center + EPSILON * np.ones((NUM_FEATURES, 1))

    # create intervals for x_1 and x_2 where x = [x_1 x_2]^T
    # the cartesian product of these
    x1_interval = np.linspace(x_min[0], x_max[0], num=DATA_PTS_PER_FEATURE)
    x2_interval = np.linspace(x_min[1], x_max[1], num=DATA_PTS_PER_FEATURE)

    # create input data matrix, which will be
    X = np.zeros((NUM_FEATURES, DATA_PTS_PER_FEATURE ** 2))
    X[0, :] = np.matlib.repmat(x1_interval, 1, DATA_PTS_PER_FEATURE).flatten()
    X[1, :] = np.kron(x2_interval, np.ones((1, DATA_PTS_PER_FEATURE))).flatten()

    return X, x_min, x_max


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

    II = np.eye(NUM_CLASSES)
    c = II[:, idx].reshape((NUM_CLASSES, 1))

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

    I_ny = np.eye(NUM_CLASSES)

    upper_outputs, lower_outputs = [], []

    for k in range(NUM_CLASSES):

        # get the k-th standard basis vector e_k = [0 ... 1 ... 0]^T
        e_k = I_ny[:, k].reshape((NUM_CLASSES, 1))

        upper_opt = solve_SDP(e_k, net, rect, verbose=True)
        lower_opt = -1. * solve_SDP(-e_k, net, rect, verbose=True)

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
    plt.plot([x1, x2], [y1, y1])
    plt.plot([x2, x2], [y1, y2])
    plt.plot([x2, x1], [y2, y2])
    plt.plot([x1, x1], [y2, y1])


if __name__ == '__main__':
    main()
