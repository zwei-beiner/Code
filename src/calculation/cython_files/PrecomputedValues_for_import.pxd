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
