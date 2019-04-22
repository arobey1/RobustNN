import numpy as np
import os
from scipy.io import savemat


class Network:

    def __init__(self, dim_list, activation, weight_type, fix_weights=False,
                    load_weights=False, weight_root=None):
        """Constructor for neural network class
        params:
            dim_list: list of ints      - dimensions of each layer in network
            actiavtion: str             - activation function for network
                                        - 'relu' | 'sigmoid' | 'tanh'
            rect: instance of Rectangle - upper and lower bounds on input
            weight_type: str            - 'ones' | 'rand'
            fix_weights: bool           - fix random seed for weights if True
        """

        # activation function for network
        self._activation = activation

        # dimensions of layers
        self._dim_list = dim_list
        self._num_hidden_layers = len(dim_list) - 2

        self._weight_type = weight_type
        self._fix_weights = fix_weights
        self._rect = None

        if load_weights == False:
            self._create_weights()
        else:
            self._weight_root = weight_root
            self._load_weights()


    @property
    def rect(self):
        return self._rect

    @rect.setter
    def rect(self, instance):
        self._rect = instance
        self._calc_lower_and_upper_bounds()

    def _create_weights(self):
        """Create weights and biases for 1-layer neural network"""

        # c is a normalizing constant for creating weight matrices
        c = 1 / np.sqrt(self._num_hidden_layers)

        self._W, self._b = {}, {}

        if self._weight_type == 'rand':

            if self._fix_weights:
                np.random.seed(5)

            for i in range(self._num_hidden_layers + 1):
                self._W[i] = c * np.random.randn(self._dim_list[i+1], self._dim_list[i])
                self._b[i] = c * np.random.randn(self._dim_list[i+1], 1)

        elif self._weight_type == 'ones':

            for i in range(self._num_hidden_layers + 1):
                self._W[i] = c * np.ones(shape=(self._dim_list[i+1], self._dim_list[i]))
                self._b[i] = c * np.ones(shape=(self._dim_list[i+1], 1))

        else:
            raise ValueError('Please use weight_type = "rand" | "ones"')

    def _load_weights(self):

        self._W, self._b = {}, {}

        weights = np.load(os.path.join(self._weight_root, 'failed_weights100.npy'), allow_pickle=True)
        biases = np.load(os.path.join(self._weight_root, 'failed_biases100.npy'), allow_pickle=True)

        for i in range(self._num_hidden_layers + 1):
            self._W[i] = weights[i]
            self._b[i] = biases[i]


    def forward_prop(self, X):
        """Compute one forward pass through the network
        params:
            X: (d,n) array of floats - training instances

        returns:
            Y: (dout,n) array of floats - output from training instances"""

        # compute for each layer in network
        # TODO: vectorize?
        Y = X
        for i in range(self._num_hidden_layers):
            Y = np.maximum(0, np.matmul(self._W[i], Y) + self._b[i])

        return np.matmul(self._W[1], Y) + self._b[1]

    def _calc_lower_and_upper_bounds(self):
        """Calculate lower and upper bounds on output of linear layers and
        output of nonlinear activation functions"""

        # lower and upper bounds for input to (nonlinear) activation function
        # for the neurons in the hidden layer
        self._x_lower_bounds = {0: self._rect.x_min}
        self._x_upper_bounds = {0: self._rect.x_max}

        # lower bound on input to hidden layer
        lower_max_term = np.matmul(np.maximum(self._W[0], 0), self._x_lower_bounds[0])
        lower_min_term = np.matmul(np.minimum(self._W[0], 0), self._x_upper_bounds[0])
        self._x_lower_bounds[1] = lower_max_term + lower_min_term + self._b[0]

        # upper bound on input to hidden layer
        upper_min_term = np.matmul(np.minimum(self._W[0], 0), self._x_lower_bounds[0])
        upper_max_term = np.matmul(np.maximum(self._W[0], 0), self._x_upper_bounds[0])
        self._x_upper_bounds[1] = upper_min_term + upper_max_term + self._b[0]

        # lower and upper bounds for output of (nonlinear) activation function
        # for the neurons in the hidden layer
        self._y_lower_bounds, self._y_upper_bounds = {}, {}

        if self._activation == 'relu':
            self._y_lower_bounds[1] = np.maximum(self._x_lower_bounds[1], 0)
            self._y_upper_bounds[1] = np.maximum(self._x_upper_bounds[1], 0)

    def save_params(self, root):

        weights = self._weights_as_list()
        biases = self._biases_as_list()

        np.save(os.path.join(root, 'failed_weights' + str(self._dim_list[1]) + '.npy'), weights)
        np.save(os.path.join(root, 'failed_biases' + str(self._dim_list[1]) + '.npy'), biases)

        savemat(os.path.join(root, 'failed_weights' + str(self._dim_list[1]) + '.npy'), mdict={'W': weights})
        savemat(os.path.join(root, 'failed_biases' + str(self._dim_list[1]) + '.npy'), mdict={'b': biases})

    def total_num_hidden_units(self):
        """Getter method for total number of hidden units in network"""

        return sum(self._dim_list[1:len(self._dim_list)-1])

    @property
    def dim_input(self):
        """Getter method for input dimension of neural network"""

        return self._dim_list[0]

    @property
    def dim_output(self):
        """Getter method for output dimension of neural network"""

        return self._dim_list[-1]

    @property
    def dim_hidden_layer(self):
        """Getter method for list of dimensions of hidden layers in neural network"""

        return self._dim_list[1]

    @property
    def W(self):
        """Getter method dictionary of weights of neural network"""

        return self._W

    @property
    def b(self):
        """Getter method dictionary of biases of neural network"""

        return self._b

    @property
    def x_lower_bounds(self):

        return self._x_lower_bounds

    @property
    def x_upper_bounds(self):

        return self._x_upper_bounds

    def _weights_as_list(self):

        return [w for w in self._W.values()]

    def _biases_as_list(self):

        return [b for b in self._b.values()]


if __name__ == '__main__':
    dim_list = [10, 20, 50, 2]
    activation = 'relu'
    net = Network(dim_list, activation, 1, 'rand')
    print([w.shape for w in net.W.values()])
