import numpy as np
import cvxpy as cvx
from scipy.linalg import block_diag
from itertools import product

from network import Network
from rectangle import Rectangle


class RobustProgram:

    def __init__(self, c, alpha, beta, net, rect):
        """Constructor for robust SDP
        params:
            nx: int - input dimension of neural network
            ny: int - output dimension of neural network
        """

        # network and rectangle instances
        self.net = net
        self.rect = rect

        # parameters
        self.num_hidden_units = self.net.total_num_hidden_units()
        self.nx, self.ny = self.net.dim_input, self.net.dim_output
        self.alpha, self.beta = alpha, beta
        self.c = c
        self.N = self.net.total_num_hidden_units()

        # define CVX variables

        # Gamma parameterizes P - QC for hyper-rectangle (cf. Prop. 1)
        self.Gamma = cvx.Variable(shape=(self.nx, self.nx), nonneg=True)

        # b paramterizes S - safety specification set (cf. Sect. 3.2)
        self.b = cvx.Variable(shape=(1, 1))

        # lam and zeta parameterize T - T paramtereizes Q (see below)
        self.lam = cvx.Variable(shape=(self.num_hidden_units, 1))
        self.zeta = cvx.Variable(shape=(self.N ** 2 - (self.N - 1), 1), nonneg=True)

        # nu and eta parameterize Q - Q parameterizes QC for ReLU function
        self.nu = cvx.Variable(shape=(self.num_hidden_units, 1), nonneg=True)
        self.eta = cvx.Variable(shape=(self.num_hidden_units, 1), nonneg=True)

        # D paramterizes M_entire
        self.D = cvx.Variable(shape=(self.nx + self.N, 1), nonneg=True)

        # create matries for objective function
        M_in = self._create_M_in_matrix()
        M_out = self._create_M_out_matrix()
        M_mid = self._create_M_mid_matrix()
        M_entire = self._create_M_entire_matrix()

        self.M_sum = M_in + M_out + M_mid + M_entire

    def _create_P_matrix(self):
        """Creates CVX variable P which characterizes the QP for hyper-rectangle
        self.Gamma is a CVX (nx, nx) variable - parameterizes P matrix

        returns:
            CVX (2nx, 2nx) variable - P decision variable in SDP
        """

        x_min, x_max = self.rect.x_min, self.rect.x_max

        P11 = -(self.Gamma + self.Gamma.T)
        P12 = (self.Gamma * x_min) + (self.Gamma.T * x_max)
        P21 = (x_min.T * self.Gamma.T) + (x_max.T * self.Gamma)
        P22 = -(x_min.T * self.Gamma.T * x_max) - (x_max.T * self.Gamma * x_min)

        return cvx.bmat([[P11, P12], [P21, P22]])

    def _create_S_matrix(self):
        """Creates CVX variable S which characterizes the safety specification set
        self.b is a CVX (1, 1) variable - b from safety set

        returns:
            CVX (2, 2) variable - S decision variable
        """

        S11 = np.zeros((self.net.dim_output, self.net.dim_output))
        S12 = self.c
        S21 = self.c.T
        S22 = -2 * self.b

        return cvx.bmat([[S11, S12], [S21, S22]])

    def _create_T_matrix(self):
        """Create CVX variable T that paramterizes the ReLU QC
        Note that a np.einsum implementation would be faster, but cvxpy does
        not allow this, so we use a for-loop at the end

        returns:
            CVX (num_hidden_units, num_hidden_units) variable - T decision variable
        """

        # below is the original numpy version - maybe it will be useful if
        # w move away from cvxpy
        # *********************************************************************
        # first_term = np.zeros((self.num_hidden_units, self.num_hidden_units))
        # np.fill_diagonal(first_term, self.lam)
        # *********************************************************************

        # first term has self.lam along the main diagonal
        first_term = cvx.diag(self.lam)

        identity = np.eye(self.num_hidden_units)
        possible_column_combos = np.array(list(product(identity, identity)))
        column_diffs = np.diff(possible_column_combos, axis=1)
        unique_col_diffs = np.squeeze(np.unique(column_diffs, axis=0))

        all_second_term_mats = np.einsum('ij,ik->ijk', unique_col_diffs, unique_col_diffs)
        (num_layers, _, _) = all_second_term_mats.shape

        # multiply second half of stack by -1 -- size of stack will be an odd
        # number and the middle entry will be the zero matrix
        all_second_term_mats[num_layers // 2 + 1:, :, :] *= -1

        # NOTE: this is the most efficient way of computing the second term
        # of T, but it seems to break np.einsum when a cvx Variable is used
        # as the second argument - maybe this is one reason to move away from cvxpy
        # *********************************************************************
        # second_term = np.einsum('ijk,i->jk', all_second_term_mats, self.zeta)
        # *********************************************************************

        # instead, we have to use a for-loop based method
        second_term = all_second_term_mats[0,:,:] * self.zeta[0]
        for i in range(1, num_layers):
            second_term += all_second_term_mats[i,:,:] * self.zeta[i]

        return first_term + second_term

    def _create_Q_matrix(self):
        """Creates CVX variable Q which characterizes the QC for ReLU
        self.nu is a (nonnegative) CVX (num_hidden_units, 1) variable
        self.eta is a (nonnegative) CVX (num_hidden_units, 1) variable
        both self.nu and self.eta parameterizer Q along with T

        returns:
            CVX (2 * num_hidden_units + 1, 2 * num_hidden_units + 1) variable
                - Q decision variable
        """

        T = self._create_T_matrix()

        Q11 = np.zeros((self.num_hidden_units, self.num_hidden_units))
        Q12 = Q21 = T
        Q13, Q31 = -self.nu, -self.nu.T
        Q22 = -2 * T
        Q23 = self.nu + self.eta
        Q32 = self.nu.T + self.eta.T
        Q33 = np.ones((1,1))

        return cvx.bmat([[Q11, Q12, Q13], [Q21, Q22, Q23], [Q31, Q32, Q33]])


    def _create_M_mid_matrix(self):
        """Create CVX variable M_mid - middle term in matrix objective function
        self.lam is a CVX (N, 1) variable - lambda from definition of T
        self.nu is a CVX (N, 1) variable  - nu from definition of QC for ReLU
        self.eta is a CVX (N, 1) variable - eta from definition of QC for ReLU

        returns:
            CVX (nx+N+1, nx+N+1) variable - M_in matrix (N is defined below)
        """

        h_dim = self.net.dim_hidden_layer
        W, b = self.net.W, self.net.b
        Q = self._create_Q_matrix()

        outer_mat = np.block([[W[0], np.zeros((h_dim, h_dim)), b[0]],
                             [np.zeros((h_dim, self.nx)), np.eye(h_dim), np.zeros((h_dim, 1))],
                             [np.zeros((1, self.nx)), np.zeros((1, h_dim)), 1]])

        return outer_mat.T * Q * outer_mat

    def _create_M_in_matrix(self):
        """Create CVX variable M_in - first term in matrix objective function
        self.Gamma is a CVX (nx, nx) variable - parameterizes P matrix

        returns:
            CVX (nx+N+1, nx+N+1) variable - M_in matrix (N is defined below)
        """

        P = self._create_P_matrix()

        outer_mat = np.block([[np.eye(self.nx), np.zeros((self.nx, self.N + 1))],
                              [np.zeros((1, self.nx + self.N)), 1]])

        return outer_mat.T * P * outer_mat

    def _create_M_out_matrix(self):
        """Create CVX variable M_out - final term in matrix objective function
        self.b is a CVX (1, 1) variable - b from safety set

        returns:
            CVX (nx+N+1, nx+N+1) variable - M_out matrix (N is defined below)
        """

        W_out, b_out = self.net.W[1], self.net.b[1]
        last_hidden_dim = self.net.dim_hidden_layer
        S = self._create_S_matrix()

        outer_mat = np.block([[np.zeros((self.ny, self.nx + self.N - last_hidden_dim)), W_out, b_out],
                              [np.zeros((1, self.nx + self.N)), 1]])

        return outer_mat.T * S * outer_mat

    def _create_M_entire_matrix(self):
        """Create CVX variable M_entire - term to account for entire network"""

        M11 = -2 * cvx.diag(self.D)

        x_min = np.vstack([val for val in self.net.x_lower_bounds.values()])
        x_max = np.vstack([val for val in self.net._x_upper_bounds.values()])

        M12 = cvx.diag(self.D) * (x_min + x_max)
        M21 = np.transpose(x_min + x_max) * cvx.diag(self.D)
        M22 = -2 * x_min.T * cvx.diag(self.D) * x_max

        return cvx.bmat([[M11, M12], [M21, M22]])

    def solve(self):
        """Solve SDP defined in the contructor of this class

        returns:
            optimal value of SDP
        """

        objective = cvx.Minimize(self.b)
        constraints = [self.M_sum << 0]

        problem = cvx.Problem(objective, constraints)
        problem.solve(verbose=True)

        return np.squeeze(self.b.value)


if __name__ == '__main__':
    dim_list = [2, 100, 2]
    ny = 2
    network = Network(dim_list, 'relu')

    xc = np.ones((2, 1))
    rect = Rectangle(xc - 0.1 * np.ones(xc.shape),
                     xc + 0.1 * np.ones(xc.shape))
    II = np.eye(ny)
    for k in range(ny):
        c = II[:, k].reshape((ny, 1))

    r = RobustProgram(c, 0, 1, network, rect)
