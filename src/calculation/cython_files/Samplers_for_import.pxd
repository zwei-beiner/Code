cdef class CategoricalSampler:
    cdef object ppf

    cpdef int sample(self, double x)


cdef class UniformSampler:
    cdef object rv

    cpdef double sample(self, double x)