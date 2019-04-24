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
ALPHA, BETA = 0, 1

def test_robustness(net, epsilon, solver='cvxopt', parallel=True):
    """Test robustness of neural network

    params:
        net: Network instance       - neural network model
        epsilon: float              - robustness parameter

    returns:
        l, u: lists of floats - result points from solving SDPs
        network_output: array - output from passing data through NN
    """

    # generated data in the rectangle defined by EPSILON
    input_data, x_lower_bound, x_upper_bound = create_rect_data(epsilon, net)
    net.rect = Rectangle(x_lower_bound, x_upper_bound)

    # pass data through neural network
    network_output = net.forward_prop(input_data)

    if parallel:
        l, u = parallel_output_rect(net, solver=solver)
    else:
        l, u = output_rect(net, solver=solver)

    return l, u, network_output

def create_rect_data(epsilon, net):
    """Create matrix of input data for neural network

    params:
        epsilon: float
        net

    returns:
        X: (in_dim, n_points ** in_dim) matrix  - input data
        x_min: (in_dim, 1) vector                     - min possible input vector
        x_max: (in_dim, 1) vector                     - max possible input vector
    """

    # point at which we test the robustness of the neural network
    x_center = np.ones((net.dim_input, 1))

    # find maximum and minimum possible deviations from x_center
    x_min = x_center - epsilon * np.ones((net.dim_input, 1))
    x_max = x_center + epsilon * np.ones((net.dim_input, 1))

    # create intervals for x_1 and x_2
    x1_interval = np.linspace(x_min[0], x_max[0], num=DATA_PTS_PER_FEATURE)
    x2_interval = np.linspace(x_min[1], x_max[1], num=DATA_PTS_PER_FEATURE)

    # create a methgrid of points
    xx, yy = np.meshgrid(x1_interval, x2_interval)

    # create input data matrix, which will be
    X = np.zeros((net.dim_input, DATA_PTS_PER_FEATURE ** 2))
    X[0,:], X[1,:] = xx.flatten(), yy.flatten()

    return X, x_min, x_max

@timing
def parallel_output_rect(net, solver='cvxopt'):
    """Solves SDPs by parallelizing over the number of features and ['upper', 'lower']

    params:
        net: Network instance       - neural network model

    returns:
        lower_outputs, upper_outputs - upper/lower points from solving SDPs
    """

    types = ['lower', 'upper']
    outputs = Parallel(n_jobs=net.dim_input * len(types))(
        delayed(parallel_solve_SDP)(k, net, b, solver=solver) for b in types for k in range(net.dim_output))

    lower_outputs = outputs[0:net.dim_input]
    upper_outputs = outputs[net.dim_input:]

    return lower_outputs, upper_outputs


def parallel_solve_SDP(idx, net, bound_type, solver='cvxopt'):
    """Called by parallel_output_rect to solve SDP for upper/lower bounds

    params:
        idx: int                    - index in dimension in output/feature space
        net: Network instance       - neural network model
        bound_type: str             - 'upper' | 'lower'
    """

    II = np.eye(net.dim_output)
    c = II[:, idx].reshape((net.dim_output, 1))

    if bound_type == 'upper':
        opt = solve_SDP(c, net, verbose=True, solver=solver)

    elif bound_type == 'lower':
        opt = -1. * solve_SDP(-c, net, verbose=True, solver=solver)

    return opt

def solve_SDP(c, net, verbose=True, solver='cvxopt'):
    """Create and solve SDP to determine support line"""

    sdp = RobustProgram(c, ALPHA, BETA, net, net.rect, solver=solver)
    solution = sdp.solve()

    if verbose:
        print('Finished solving SDP')
        print('Solution:')
        print(solution)
        print('-' * 50 + '\n')

    return solution


@timing
def output_rect(net, solver='cvxopt'):
    """Find upper and lower bounding lines
    params:
        net: Network instance       - neural network

    returns:
        l, u: lists of floats - lower and upper bounds
    """

    I_ny = np.eye(net.dim_output)

    upper_outputs, lower_outputs = [], []

    for k in range(net.dim_output):

        # get the k-th standard basis vector e_k = [0 ... 1 ... 0]^T
        e_k = I_ny[:, k].reshape((net.dim_output, 1))

        upper_opt = solve_SDP(e_k, net, verbose=True, solver=solver)
        lower_opt = -1. * solve_SDP(-e_k, net, verbose=True, solver=solver)

        upper_outputs.append(upper_opt)
        lower_outputs.append(lower_opt)

    return lower_outputs, upper_outputs



# if __name__ == '__main__':
#     main()
