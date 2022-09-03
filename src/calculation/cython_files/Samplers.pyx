cimport numpy as np
import numpy as np

import scipy.stats
from pypolychord.priors import UniformPrior


cdef class CategoricalSampler:
    cdef object ppf

    def __init__(self, int length):
        """
        Creates function which maps the continuous random variable Uniform([0, 1]) to a discrete uniform random variable
        over {0, 1, ..., len(categories) - 1}.
        """

        rv = scipy.stats.randint(low=0, high=length)
        self.ppf = rv.ppf

    cpdef int sample(self, double x):
        # x=0 has to be handled separately.
        # (See https://stackoverflow.com/questions/25688461/ppf0-of-scipys-randint0-2-is-1-0)
        if x == 0.0:
            return 0
        # If x is not in [0, 1] (interval includes endpoints), returns np.nan.
        # np.nan is not handled here because it is assumed that the input is in the range [0, 1].
        index = self.ppf(x)
        return np.int_(index)


cdef class UniformSampler:
    cdef object rv

    def __init__(self, double min, double max):
        self.rv = UniformPrior(min, max)

    cpdef double sample(self, double x):
        return self.rv(x)