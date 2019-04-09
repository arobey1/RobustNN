import numpy as np


class Network:

    def __init__(self, dim_list, activation, rect, weight_type):
        """Constructor for neural network class
        params:
            dim_list: list of ints      - dimensions of each layer in network
            actiavtion: str             - activation function for network
                                        - 'relu' | 'sigmoid' | 'tanh'
            rect: instance of Rectangle - upper and lower bounds on input
            weight_type: str            - 'ones' | 'rand'
        """

        # activation function for network
        self._activation = activation

        # dimensions of layers
        self._dim_input, self._dim_output = dim_list[0], dim_list[2]
        self._dim_hidden_layer = dim_list[1]
        self._num_hidden_layers = 1

        self._weight_type = weight_type
        self._rect = rect

        self._create_weights()
        self._calc_lower_and_upper_bounds()

        print(self._W[0])

    def _create_weights(self):
        """Create weights and biases for 1-layer neural network"""

        # c is a normalizing constant for creating weight matrices
        c = 1 / np.sqrt(self._num_hidden_layers)

        self._W, self._b = {}, {}

        if self._weight_type == 'rand':
            self._W[0] = c * np.random.randn(self._dim_hidden_layer, self._dim_input)
            self._b[0] = c * np.random.randn(self._dim_hidden_layer, 1)

            self._W[1] = c * np.random.randn(self._dim_output, self._dim_hidden_layer)
            self._b[1] = c * np.random.randn(self._dim_output, 1)

        elif self._weight_type == 'ones':
            self._W[0] = c * np.ones(shape=(self._dim_hidden_layer, self._dim_input))
            self._b[0] = c * np.ones(shape=(self._dim_hidden_layer, 1))

            self._W[1] = c * np.ones(shape=(self._dim_output, self._dim_hidden_layer))
            self._b[1] = c * np.ones(shape=(self._dim_output, 1))

        else:
            raise ValueError('Please use weight_type = "rand" | "ones"')

    def forward_prop(self, X):
        """Compute one forward pass through the network
        params:
            X: d x n array of floats - training instances

        returns:
            Y: dout x n array of floats - output from training instances"""

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

    def total_num_hidden_units(self):
        """Getter method for total number of hidden units in network"""

        return self._dim_hidden_layer

    @property
    def dim_input(self):
        """Getter method for input dimension of neural network"""

        return self._dim_input

    @property
    def dim_output(self):
        """Getter method for output dimension of neural network"""

        return self._dim_output

    @property
    def dim_hidden_layer(self):
        """Getter method for list of dimensions of hidden layers in neural network"""

        return self._dim_hidden_layer

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

    # @lower_bounds.setter
    # def lower_bounds(self, key, value):
    #     """Setting method dictionary of lower bounds of neural network"""
    #
    #     self._lower_bounds[key] = value
    #
    # @upper_bounds.setter
    # def upper_bounds(self, key, value):
    #     """Setting method dictionary of upper bounds of neural network"""
    #
    #     self._upper_bounds[key] = value


if __name__ == '__main__':
    dim_list = [2, 20, 50, 2]
    activation = 'relu'
    n = Network(dim_list, activation)
    print(n.dim_hidden[-1])
