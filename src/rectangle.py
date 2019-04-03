class Rectangle:
    def __init__(self, x_min, x_max):
        """Constructor for rectangle class
        params:
            x_min: (2, 1) matrix of floats - positive deviation from central pt xc
            x_max: (2, 1) matrix of floats - negative deviation from central pt xc
        """

        self._x_min = x_min
        self._x_max = x_max

    @property
    def x_min(self):
        """Getter method for x_min"""

        return self._x_min

    @property
    def x_max(self):
        """Getter method for x_max"""

        return self._x_max
