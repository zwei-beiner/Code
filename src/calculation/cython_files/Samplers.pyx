cimport numpy as np
import numpy as np

from pypolychord.priors import UniformPrior


cdef class CategoricalSampler:
    cdef double length

    def __init__(self, int length):
        """
        Creates function which maps the continuous random variable Uniform([0, 1]) to a discrete uniform random variable
        over {0, 1, ..., len(categories) - 1}.
        """

        self.length = np.float64(length)


    cpdef double sample(self, double x):
        if x == 1.:
            return self.length - 1.
        return np.floor(self.length * x)


cdef class UniformSampler:
    cdef object rv

    def __init__(self, double min, double max):
        self.rv = UniformPrior(min, max)

    cpdef double sample(self, double x):
        return self.rv(x)