# Note: The functions here should implement an abstract class, e.g.
#
# cdef class Sampler:
#   cpdef double sample(self, double x)
#
# However, Cython does not support abstract classes. Therefore, the programmer must ensure that this interface
# is implemented by all the sampler classes below.


cimport numpy as np
import numpy as np

from pypolychord.priors import UniformPrior


cdef class CategoricalSampler:
    """
    Creates function which maps the continuous random variable Uniform([0, 1]) to a discrete uniform random variable
    over {0, 1, ..., len(categories) - 1}.
    """

    cdef double length

    def __init__(self, int length):
        self.length = np.float64(length)

    cpdef double sample(self, double x):
        """
        @param x: A double in the range [0, 1]
        @return: An integer (cast to a double) in the range {0, 1, ..., self.length - 1}
        """

        # Technically, the input x=1. has probability zero of occurring, but in case that it happens, a valid array
        # index must be returned. If one would simply floor x, then 'length' would be returned, which is not valid
        # output.
        if x == 1.:
            return self.length - 1.
        # Use the trick of multiplying x by self.length and flooring the result.
        return np.floor(self.length * x)


cdef class UniformSampler:
    """
    Creates a function which maps the continuous random variable Uniform([0, 1]) to a uniform random variable over
    [min, max].
    """

    # Python Callable of type Callable[[float], float]
    cdef object rv

    def __init__(self, double min, double max):
        # Use the prior provided by PolyChord
        self.rv = UniformPrior(min, max)

    cpdef double sample(self, double x):
        """
        @param x: A double in the range [0, 1]
        @return: A double in the range [min, max]
        """
        return self.rv(x)