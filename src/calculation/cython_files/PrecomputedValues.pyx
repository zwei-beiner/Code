cimport numpy as np
import numpy as np

cdef class PrecomputedValues:
    """
    Class which stores precomputed values for a set of wavelengths.
    Essentially, this means that all functions of wavelength are turned into arrays which store the function
    values at each wavelength.
    """

    # Number of layers
    cdef int M
    # Number of wavelengths. This determines the length of the arrays used.
    cdef int num_wavelengths

    # Store n_outer at each wavelength
    cdef double[:] n_outer
    # Store n_substrate at each wavelength
    cdef double[:] n_substrate
    # Wavelengths
    cdef double[:] wavelengths
    # Store refractive index arrays as 2D array, and each entry an array n(wavelength).
    # First index: which layer?. Second index: which refractive index?
    cdef object[:] ns

    cdef double p_pol_weighting
    cdef double s_pol_weighting
    cdef double sum_weighting
    cdef double difference_weighting
    cdef double phase_weighting

    # Store the parameters of the merit function for all wavelengths.
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
        @param M: Number of layers
        @param n_outer: Type is Callable[[np.ndarray], np.ndarray]
        @param n_substrate: Type is Callable[[np.ndarray], np.ndarray]
        @param wavelengths: Array of wavelengths
        @param layer_specification: Type is tuple[list[RefractiveIndex]]
        @param m: Type is MeritFunctionSpecification
        """

        if M != len(layer_specification):
            raise Exception(f'{M} != {len(layer_specification)}')

        self.M = M
        self.wavelengths = wavelengths
        self.num_wavelengths = len(wavelengths)
        # Evaluate n_outer and n_substrate at the wavelengths.
        self.n_outer = n_outer(wavelengths)
        self.n_substrate = n_substrate(wavelengths)

        # Example for 'layer_specification':
        # ([lambda x: np.full(len(x), 1.3)], [lambda x: np.full(len(x), 2, dtype=np.float_), lambda x: np.full(len(x), 3.), lambda x: np.full(len(x), 4.)])
        # Store an array of numpy arrays. This is not a 2D numpy array because the 1D arrays have different lengths
        # because the sets of wavelengths have different numbers of wavelengths in them.
        self.ns = np.empty((M, ), dtype=object)
        cdef int i
        for i in range(len(layer_specification)):
            layer = layer_specification[i]
            self.ns[i] = np.array([n(np.asarray(self.wavelengths)) for n in layer], dtype=object)

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
