cimport numpy as np
import numpy as np

cdef class PrecomputedValues:
    cdef int M
    cdef int num_wavelengths

    cdef double[:] n_outer
    cdef double[:] n_substrate
    cdef double[:] wavelengths
    # Store refractive index arrays as 2D array, and each entry an array n(wavelength).
    # First index: which layer?. Second index: which refractive index?
    cdef object[:] ns

    cdef double p_pol_weighting
    cdef double s_pol_weighting
    cdef double sum_weighting
    cdef double difference_weighting
    cdef double phase_weighting

    cdef double[:] target_reflectivity_s
    cdef double[:] target_reflectivity_p
    cdef double[:] target_sum
    cdef double[:] target_difference
    cdef double[:] target_relative_phase
    cdef double[:] weight_function_s
    cdef double[:] weight_function_p
    cdef double[:] weight_function_sum
    cdef double[:] weight_function_difference
    cdef double[:] weight_function_phase

    def __init__(self, int M, object n_outer, object n_substrate, double[:] wavelengths, object layer_specification, object m):
        """

        @param M:
        @param n_outer: Type is Callable[[np.ndarray], np.ndarray]
        @param n_substrate: Type is Callable[[np.ndarray], np.ndarray]
        @param wavelengths:
        @param layer_specification:
        @param m: Type is MeritFunctionSpecification
        """

        if M != len(layer_specification):
            raise Exception(f'{M} != {len(layer_specification)}')

        self.M = M
        self.wavelengths = wavelengths
        self.num_wavelengths = len(wavelengths)
        self.n_outer = n_outer(wavelengths)
        self.n_substrate = n_substrate(wavelengths)

        # Example layer specification:
        # ([lambda x: np.full(len(x), 1.3)], [lambda x: np.full(len(x), 2, dtype=np.float_), lambda x: np.full(len(x), 3.), lambda x: np.full(len(x), 4.)])
        self.ns = np.empty((M, ), dtype=object)
        cdef int i
        for i in range(len(layer_specification)):
            layer = layer_specification[i]
            self.ns[i] = np.array([n(self.wavelengths) for n in layer], dtype=object)

        self.p_pol_weighting = m.p_pol_weighting
        self.s_pol_weighting = m.s_pol_weighting
        self.sum_weighting = m.sum_weighting
        self.difference_weighting = m.difference_weighting
        self.phase_weighting = m.phase_weighting

        self.target_reflectivity_s = m.target_reflectivity_s(wavelengths)
        self.target_reflectivity_p = m.target_reflectivity_p(wavelengths)
        self.target_sum = m.target_sum(wavelengths)
        self.target_difference = m.target_difference(wavelengths)
        self.target_relative_phase = m.target_relative_phase(wavelengths)
        self.weight_function_s = m.weight_function_s(wavelengths)
        self.weight_function_p = m.weight_function_p(wavelengths)
        self.weight_function_sum = m.weight_function_sum(wavelengths)
        self.weight_function_difference = m.weight_function_difference(wavelengths)
        self.weight_function_phase = m.weight_function_phase(wavelengths)

    def __str__(self):
        return f'{self.num_wavelengths}'


    # cpdef np.ndarray [double, ndim=2] ns_at_optimal_indices(self, np.ndarray [long, ndim=1] optimal_n):
    #     cdef np.ndarray [double, ndim=2] ns = np.zeros((self.M, self.num_wavelengths))
    #
    #     cdef int i
    #     for i in range(self.M):
    #         ns[i, :] = self.ns[i][optimal_n[i]]
    #
    #     return ns