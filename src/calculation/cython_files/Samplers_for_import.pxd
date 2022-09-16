cdef class CategoricalSampler:
    cdef double length

    cpdef double sample(self, double x)


cdef class UniformSampler:
    cdef object rv

    cpdef double sample(self, double x)