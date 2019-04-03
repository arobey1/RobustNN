import numpy as np


class Network:

    def __init__(self, dim_list, activation):
        """Constructor for neural network class
        params:
            dim_list: list of ints    - dimensions of each layer in network
            actiavtion: str         - activation function for network
                                    - 'relu' | 'sigmoid' | 'tanh'
        """

        # activation function for network
        self._activation = activation

        # dimensions of layers
        self._dim_in, self._dim_out = dim_list[0], dim_list[-1]
        self._dim_hidden = dim_list[1: len(dim_list) - 1]

        self._num_hidden_layers = len(dim_list) - 2
        c = 1 / np.sqrt(self._num_hidden_layers)

        # dictionaries for weights and biases of network
        self._W, self._b = {}, {}
        for i in range(self._num_hidden_layers):
            self._W[i] = c * np.random.randn(dim_list[i + 1], dim_list[i])
            self._b[i] = c * np.random.randn(dim_list[i + 1], 1)

        # weights and biases for final layer
        self._W_out = c * np.random.randn(self._dim_out, dim_list[-2])
        self._b_out = c * np.random.randn(self._dim_out, 1)

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

        return np.matmul(self._W_out, Y) + self._b_out

    def total_num_hidden_units(self):
        """Getter method for total number of hidden units in network"""

        return sum(self._dim_hidden)

    @property
    def num_hidden_layers(self):
        """Getter method for number of hidden layers of neural network"""

        return self._num_hidden_layers

    @property
    def dim_in(self):
        """Getter method for input dimension of neural network"""

        return self._dim_in

    @property
    def dim_out(self):
        """Getter method for output dimension of neural network"""

        return self._dim_out

    @property
    def W_out(self):
        """Getter method for weights of last layer of neural network"""

        return self._W_out

    @property
    def b_out(self):
        """Getter method for biases of last layer of neural network"""

        return self._b_out

    @property
    def dim_hidden(self):
        """Getter method for list of dimensions of hidden layers in neural network"""

        return self._dim_hidden

    @property
    def W(self):
        """Getter method dictionary of weights of neural network"""

        return self._W

    @property
    def b(self):
        """Getter method dictionary of biases of neural network"""

        return self._b


if __name__ == '__main__':
    dim_list = [2, 20, 50, 2]
    activation = 'relu'
    n = Network(dim_list, activation)
    print(n.dim_hidden[-1])
