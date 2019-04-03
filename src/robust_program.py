import numpy as np
import cvxpy as cvx

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
        self.total_hidden_units = self.net.total_num_hidden_units()
        self.nx, self.ny = self.net.dim_in, self.net.dim_in
        self.alpha, self.beta = alpha, beta
        self.c = c

        # define CVX variables
        self.tau = cvx.Variable(shape=(self.nx, self.nx))
        self.b_i = cvx.Variable(shape=(1, 1))
        self.lam = cvx.Variable(shape=(self.total_hidden_units, 1))
        self.nu = cvx.Variable(shape=(self.total_hidden_units, 1))
        self.eta = cvx.Variable(shape=(self.total_hidden_units, 1))

        # create matries for objective function
        M_in = self.__create_M_in_matrix()
        M_out = self.__create_M_out_matrix()
        M_mid = self.__create_M_mid_matrix()
        self.M_sum = M_in + M_out + M_mid

    def __create_P_matrix(self):
        """Creates CVX variable P which characterizes the QP for hyper-rectangle
        self.tau is a CVX (nx, nx) variable - parameterizes P matrix

        returns:
            CVX (2nx, 2nx) variable - P decision variable in SDP
        """

        x_min, x_max = self.rect.x_min, self.rect.x_max

        P11 = -(self.tau + self.tau.T)
        P12 = (self.tau * x_min) + (self.tau.T * x_max)
        P21 = (x_min.T * self.tau.T) + (x_max.T * self.tau)
        P22 = -(x_min.T * self.tau.T * x_max) - (x_max.T * self.tau * x_min)

        return cvx.bmat([[P11, P12], [P21, P22]])

    def __create_S_matrix(self):
        """Creates CVX variable S which characterizes the safety specification set
        self.b_i is a CVX (1, 1) variable - b from safety set

        returns:
            CVX (2, 2) variable - S decision variable
        """

        S11 = np.zeros((self.ny, self.ny))
        S12 = self.c
        S21 = self.c.T
        S22 = -2 * self.b_i

        return cvx.bmat([[S11, S12], [S21, S22]])

    def __create_M_mid_matrix(self):
        """Create CVX variable M_mid - middle term in matrix objective function
        self.lam is a CVX (N, 1) variable - lambda from definition of T
        self.nu is a CVX (N, 1) variable  - nu from definition of QC for ReLU
        self.eta is a CVX (N, 1) variable - eta from definition of QC for ReLU

        returns:
            CVX (nx+N+1, nx+N+1) variable - M_in matrix (N is defined below)
        """

        T = np.zeros((self.total_hidden_units, self.total_hidden_units))
        N = self.net.total_num_hidden_units()
        last_hidden_dim = self.net.dim_hidden[-1]

        # get weights and biases
        W, b = self.net.W[0], self.net.b[0]
        W = np.concatenate((W, np.zeros((N, last_hidden_dim))), axis=1)
        B = np.zeros((N, self.net.dim_in))
        B = np.concatenate((B, np.eye(N)), axis=1)

        # create necessary components of Q matrix
        Q11 = -2 * self.alpha * self.beta * (cvx.diag(self.lam) + T)
        Q12 = (self.alpha + self.beta) * (cvx.diag(self.lam) + T)
        Q13 = -self.nu
        Q21 = Q12.T
        Q22 = -2 * (cvx.diag(self.lam)) - (T + T.T)
        Q23 = self.eta + self.nu
        Q31 = Q13.T
        Q33 = 0

        # now create components of N_mid
        tmp = (B.T) * Q21 * W
        M11 = (W.T * Q11 * W) + (tmp + tmp.T) + (B.T * Q22 * B)

        M12 = (W.T * Q11 * b) + (B.T * Q21 * b) + (W.T * Q13) + (B.T * Q23)
        M21 = M12.T

        tmp = Q31 * b
        M22 = (b.T * Q11 * b) + (tmp + tmp.T) + Q33

        return cvx.bmat([[M11, M12], [M21, M22]])

    def __create_M_in_matrix(self):
        """Create CVX variable M_in - first term in matrix objective function
        self.tau is a CVX (nx, nx) variable - parameterizes P matrix

        returns:
            CVX (nx+N+1, nx+N+1) variable - M_in matrix (N is defined below)
        """

        N = self.net.total_num_hidden_units()
        P = self.__create_P_matrix()

        outer_mat = np.block([[np.eye(self.nx), np.zeros((self.nx, N + 1))],
                              [np.zeros((1, self.nx + N)), 1]])

        return outer_mat.T * P * outer_mat

    def __create_M_out_matrix(self):
        """Create CVX variable M_out - final term in matrix objective function
        self.b_i is a CVX (1, 1) variable - b from safety set

        returns:
            CVX (nx+N+1, nx+N+1) variable - M_out matrix (N is defined below)
        """

        N = self.net.total_num_hidden_units()
        W_out, b_out = self.net.W_out, self.net.b_out
        last_hidden_dim = self.net.dim_hidden[-1]
        S = self.__create_S_matrix()

        outer_mat = np.block([[np.zeros((self.ny, self.nx + N - last_hidden_dim)), W_out, b_out],
                              [np.zeros((1, self.nx + N)), 1]])

        return outer_mat.T * S * outer_mat

    def solve(self):
        """Solve SDP defined in the contructor of this class
        returns:
            optimal value of SDP
        """

        objective = cvx.Minimize(self.b_i)

        # add constraints to problem
        constraints = [self.M_sum << 0]
        for i in range(self.nx):
            for j in range(self.ny):
                constraints.append(self.tau[i, j] >= 0)
        constraints.append(self.nu >= 0)
        constraints.append(self.eta >= 0)

        # define and solve problem
        problem = cvx.Problem(objective, constraints)
        problem.solve(verbose=True)

        return np.squeeze(self.b_i.value)


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
