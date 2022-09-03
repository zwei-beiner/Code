import sys

cimport numpy as np
import numpy as np
import inspect


# from MeritFunctionSpecification import MeritFunctionSpecification
# import sys
# sys.path.insert(1, '/Users/namu/Desktop/Part_II/Summer_Project/Code/src/calculation/reflectivity.pxd')
# print(sys.path)
# from reflectivity cimport calculate_wavelengths, amplitude, amplitude_at_wavelengths
from reflectivity_for_import cimport reflectivity_namespace
from Samplers_for_import cimport CategoricalSampler, UniformSampler
from PrecomputedValues_for_import cimport PrecomputedValues


cdef class BackendCalculations:
    cdef long M
    cdef double theta_outer

    # Array of type PrecomputedValues. Store automatically calculated wavelengths.
    cdef object[:] precomputed_values_array
    # Store wavelengths for plotting.
    cdef object precomputed_values_for_plotting
    #  Store wavelengths manually specified by user.
    cdef object precomputed_values_manually_specified

    cdef double delta_D
    cdef int K_max

    cdef long[:] d_fixed_indices
    cdef long[:] d_unfixed_indices
    cdef double[:] d_fixed_values
    cdef long[:] n_fixed_indices
    cdef long[:] n_unfixed_indices

    cdef long split
    cdef long nDims
    cdef double d_fixed_sum

    cdef object[:] samplers

    # Int array which acts as a boolean array and stores whether weighting factors are zero.
    # Used in merit function calculation.
    # If entry is 1, interpret as True. If entry is 0, interpret as false.
    # Note: An int array must be used because boolean memoryviews are not supported in Cython.
    # (See https://stackoverflow.com/questions/49058191/boolean-numpy-arrays-with-cython)
    cdef int[:] is_term_switched_on

    cdef object r

    def __init__(self, long M, object n_outer, object n_substrate, double theta_outer, double D_max, double min_wavelength, double max_wavelength,
                 object layer_specification, object merit_function_specification,
                 long[:] d_fixed_indices, long[:] d_unfixed_indices, double[:] d_fixed_values, object d_unfixed_values,
                 long[:] n_fixed_indices, long[:] n_unfixed_indices, object n_unfixed_values,
                 long split, long nDims,
                 custom_wavelengths=None):
        """

        @param M:
        @param theta_outer:
        @param D_max:
        @param min_wavelength:
        @param max_wavelength:
        @param layer_specification: Type is tuple[list[RefractiveIndex]]
        @param merit_function_specification: Type is MeritFunctionSpecification
        @param d_fixed_indices:
        @param d_unfixed_indices:
        @param d_fixed_values:
        @param d_unfixed_values: Type is list[tuple[float, float]]
        @param n_fixed_indices:
        @param n_unfixed_indices:
        @param n_unfixed_values: Type is list[list[RefractiveIndex]]
        @param split:
        @param custom_wavelengths:
        """
        self.r = reflectivity_namespace()

        self.M = M
        self.theta_outer = theta_outer

        self.d_fixed_indices = d_fixed_indices
        self.d_unfixed_indices = d_unfixed_indices
        self.d_fixed_values = d_fixed_values
        self.n_fixed_indices = n_fixed_indices
        self.n_unfixed_indices = n_unfixed_indices

        self.split = split
        self.nDims = nDims

        cdef double const = 8. * (1. / min_wavelength - 1. / max_wavelength)
        # K_max is chosen such that all allowed values of K that can be computed are in [0,...,K_max]
        # (0 and K_max inclusive).
        self.K_max = np.floor(D_max * const) + 1
        # print(K_max)

        # Fill up precomputed_values_array.
        self.precomputed_values_array = np.empty((self.K_max, ), dtype=object)
        self.delta_D = 1. / const
        cdef int i
        cdef double[:] wavelengths
        for i in range(self.K_max):
            # Introduce offset 1.5 in calculation of D to get a non-zero D at i=0.
            # print('K:', i)
            wavelengths = self.r.calculate_wavelengths(min_wavelength, max_wavelength, (i + 1.5) * self.delta_D)
            self.precomputed_values_array[i] = PrecomputedValues(self.M, n_outer, n_substrate, wavelengths, layer_specification, merit_function_specification)

        self.precomputed_values_for_plotting = PrecomputedValues(self.M, n_outer, n_substrate, np.linspace(min_wavelength, max_wavelength, num=1000), layer_specification, merit_function_specification)
        if custom_wavelengths is not None:
            self.precomputed_values_manually_specified = PrecomputedValues(self.M, n_outer, n_substrate, custom_wavelengths, layer_specification, merit_function_specification)

        self.d_fixed_sum = np.sum(d_fixed_values)

        # Pre-allocate samplers.
        self.samplers = np.empty(self.nDims, dtype=object)
        for i in range(self.nDims):
            if i < self.split:
                self.samplers[i] = CategoricalSampler(len(n_unfixed_values[i]))
            else:
                self.samplers[i] = UniformSampler(d_unfixed_values[i - self.split][0], d_unfixed_values[i - self.split][1])


        self.is_term_switched_on = np.zeros(4, dtype=np.int32)
        self.is_term_switched_on[0] = 1 if (merit_function_specification.s_pol_weighting == 0.0) else 0
        self.is_term_switched_on[1] = 1 if (merit_function_specification.p_pol_weighting == 0.0) else 0
        self.is_term_switched_on[2] = 1 if (merit_function_specification.sum_weighting == 0.0) else 0
        self.is_term_switched_on[3] = 1 if (merit_function_specification.difference_weighting == 0.0) else 0
        self.is_term_switched_on[4] = 1 if (merit_function_specification.phase_weighting == 0.0) else 0

        # samplers = [Utils.categorical_sampler(n_unfixed_values[i]) for i in range(0, split)] + [
        #     Utils.uniform_sampler(*d_unfixed_values[i]) for i in range(0, nDims - split)]

        # self.precomputed_values_array = np.array([PrecomputedValues(self.M, np.linspace(1, 10, num=4, dtype=np.float_), layer_specification) for i in range(4)], dtype=object)
        # print('wavelengths', np.asarray(self.precomputed_values_array[2].wavelengths))


    cpdef void amplitude_wrapper(self, PrecomputedValues precomputed_values, np.ndarray [long, ndim=1] n_params,
                                 np.ndarray [double, ndim=1] d_params, np.ndarray [complex, ndim=1] amplitude_s,
                                 np.ndarray [complex, ndim=1] amplitude_p):
        cdef int i, j
        cdef double wavelength
        cdef double n_outer
        cdef double n_substrate
        cdef int fixed_index
        cdef int unfixed_index

        # Pre-allocate arrays with the appropriate size.
        n = np.zeros(self.M, dtype=np.float_)
        d = np.zeros(self.M, dtype=np.float_)
        # Already fill indices where d is fixed.
        for i in range(len(self.d_fixed_indices)):
            d[self.d_fixed_indices[i]] = self.d_fixed_values[i]

        # Fill up the amplitude_s and amplitude_p arrays.
        for i in range(len(precomputed_values.wavelengths)):
            wavelength = precomputed_values.wavelengths[i]

            n_outer = precomputed_values.n_outer[i]
            n_substrate = precomputed_values.n_substrate[i]

            # Fill up n array.
            for j in range(len(self.n_fixed_indices)):
                fixed_index = self.n_fixed_indices[j]
                n[fixed_index] = precomputed_values.ns[fixed_index][0][i]

            for j in range(len(self.n_unfixed_indices)):
                unfixed_index = self.n_unfixed_indices[j]
                n[unfixed_index] = precomputed_values.ns[unfixed_index][n_params[j]][i]

            # Fill up d array.
            for j in range(len(self.d_unfixed_indices)):
                unfixed_index = self.d_unfixed_indices[j]
                d[unfixed_index] = d_params[j]

            # Calculate amplitudes
            amplitude_s[i] = self.r.amplitude(0, self.M, n, d, wavelength, n_outer, n_substrate, self.theta_outer)
            amplitude_p[i] = self.r.amplitude(1, self.M, n, d, wavelength, n_outer, n_substrate, self.theta_outer)


    cpdef double merit_function_auto_wavelength(self, np.ndarray [double, ndim=1] params):
        cdef np.ndarray [long, ndim=1] n_params = np.int_(params[:self.split])
        cdef np.ndarray [double, ndim=1] d_params = params[self.split:]

        cdef double total_thickness = d_params.sum() + self.d_fixed_sum
        cdef int K = np.floor(total_thickness / self.delta_D)

        cdef PrecomputedValues p = self.precomputed_values_array[K]
        cdef int num_wavelengths = len(p.wavelengths)
        cdef np.ndarray [complex, ndim=1] amplitude_s = np.zeros(num_wavelengths, dtype=complex)
        cdef np.ndarray [complex, ndim=1] amplitude_p = np.zeros(num_wavelengths, dtype=complex)

        # Fill amplitude_s and amplitde_p arrays.
        self.amplitude_wrapper(p, n_params, d_params, amplitude_s, amplitude_p)

        cdef double res = 0
        cdef np.ndarray[double, ndim=1] reflectivity_s
        cdef np.ndarray[double, ndim=1] reflectivity_p
        cdef np.ndarray[double, ndim=1] sum_of_pol
        cdef np.ndarray[double, ndim=1] diff_of_pol
        cdef np.ndarray[double, ndim=1] relative_phases
        if self.is_term_switched_on[0] == 0:
            reflectivity_s = np.abs(amplitude_s) ** 2
            res = res + p.s_pol_weighting * np.mean(
                ((reflectivity_s - p.target_reflectivity_s) / p.weight_function_s) ** 2)
        if self.is_term_switched_on[1] == 0:
            reflectivity_p = np.abs(amplitude_p) ** 2
            res = res + p.p_pol_weighting * np.mean(
                ((reflectivity_p - p.target_reflectivity_p) / p.weight_function_p) ** 2)
        if self.is_term_switched_on[2] == 0:
            sum_of_pol = np.abs(np.abs(amplitude_s) ** 2 + np.abs(amplitude_p) ** 2)
            res = res + p.sum_weighting * np.mean(
                ((sum_of_pol - p.target_sum) / p.weight_function_sum) ** 2)
        if self.is_term_switched_on[3] == 0:
            diff_of_pol = np.abs(np.abs(amplitude_s) ** 2 - np.abs(amplitude_p) ** 2)
            res = res + p.difference_weighting * np.mean(
                ((diff_of_pol - p.target_difference) / p.weight_function_difference) ** 2)
        if self.is_term_switched_on[4] == 0:
            relative_phases = np.angle(amplitude_s / amplitude_p)
            res = res + p.phase_weighting * np.mean(
                ((relative_phases - p.target_relative_phase) / p.weight_function_phase) ** 2)

        return res

        # cdef np.ndarray [double, ndim=1] reflectivity_s = np.abs(amplitude_s) ** 2
        # cdef np.ndarray [double, ndim=1] reflectivity_p = np.abs(amplitude_p) ** 2
        # cdef np.ndarray [double, ndim=1] relative_phases = np.angle(amplitude_s / amplitude_p)
        # cdef np.ndarray [double, ndim=1] sum_of_pol = np.abs(reflectivity_s + reflectivity_p)
        # cdef np.ndarray [double, ndim=1] diff_of_pol = np.abs(reflectivity_s - reflectivity_p)
        #
        # return p.s_pol_weighting * np.mean(((reflectivity_s - p.target_reflectivity_s) / p.weight_function_s) ** 2) + \
        #        p.p_pol_weighting * np.mean(((reflectivity_p - p.target_reflectivity_p) / p.weight_function_p) ** 2) + \
        #        p.sum_weighting * np.mean(((sum_of_pol - p.target_sum) / p.weight_function_sum) ** 2) + \
        #        p.difference_weighting * np.mean(((diff_of_pol - p.target_difference) / p.weight_function_difference) ** 2) + \
        #        p.phase_weighting * np.mean(((relative_phases - p.target_relative_phase) / p.weight_function_phase) ** 2)


    cpdef double merit_function_fixed_wavelength(self, np.ndarray [double, ndim=1] params):
        """
        Same function as 'self.merit_function_auto_wavelength()' but using the user-defined wavelengths. 
        """

        cdef np.ndarray[long, ndim=1] n_params = np.int_(params[:self.split])
        cdef np.ndarray[double, ndim=1] d_params = params[self.split:]

        cdef PrecomputedValues p = self.precomputed_values_manually_specified
        cdef int num_wavelengths = len(p.wavelengths)
        cdef np.ndarray[complex, ndim=1] amplitude_s = np.zeros(num_wavelengths, dtype=complex)
        cdef np.ndarray[complex, ndim=1] amplitude_p = np.zeros(num_wavelengths, dtype=complex)

        # Fill amplitude_s and amplitde_p arrays.
        self.amplitude_wrapper(p, n_params, d_params, amplitude_s, amplitude_p)

        cdef double res = 0
        cdef np.ndarray[double, ndim=1] reflectivity_s
        cdef np.ndarray[double, ndim=1] reflectivity_p
        cdef np.ndarray[double, ndim=1] sum_of_pol
        cdef np.ndarray[double, ndim=1] diff_of_pol
        cdef np.ndarray[double, ndim=1] relative_phases
        if self.is_term_switched_on[0] == 0:
            reflectivity_s = np.abs(amplitude_s) ** 2
            res = res + p.s_pol_weighting * np.mean(
                ((reflectivity_s - p.target_reflectivity_s) / p.weight_function_s) ** 2)
        if self.is_term_switched_on[1] == 0:
            reflectivity_p = np.abs(amplitude_p) ** 2
            res = res + p.p_pol_weighting * np.mean(
                ((reflectivity_p - p.target_reflectivity_p) / p.weight_function_p) ** 2)
        if self.is_term_switched_on[2] == 0:
            sum_of_pol = np.abs(np.abs(amplitude_s) ** 2 + np.abs(amplitude_p) ** 2)
            res = res + p.sum_weighting * np.mean(
                ((sum_of_pol - p.target_sum) / p.weight_function_sum) ** 2)
        if self.is_term_switched_on[3] == 0:
            diff_of_pol = np.abs(np.abs(amplitude_s) ** 2 - np.abs(amplitude_p) ** 2)
            res = res + p.difference_weighting * np.mean(
                ((diff_of_pol - p.target_difference) / p.weight_function_difference) ** 2)
        if self.is_term_switched_on[4] == 0:
            relative_phases = np.angle(amplitude_s / amplitude_p)
            res = res + p.phase_weighting * np.mean(
                ((relative_phases - p.target_relative_phase) / p.weight_function_phase) ** 2)

        return p.s_pol_weighting * np.mean(((reflectivity_s - p.target_reflectivity_s) / p.weight_function_s) ** 2) + \
               p.p_pol_weighting * np.mean(((reflectivity_p - p.target_reflectivity_p) / p.weight_function_p) ** 2) + \
               p.sum_weighting * np.mean(((sum_of_pol - p.target_sum) / p.weight_function_sum) ** 2) + \
               p.difference_weighting * np.mean(
            ((diff_of_pol - p.target_difference) / p.weight_function_difference) ** 2) + \
               p.phase_weighting * np.mean(((relative_phases - p.target_relative_phase) / p.weight_function_phase) ** 2)


    cpdef np.ndarray [double, ndim=1] prior(self, np.ndarray [double, ndim=1] unit_cube):
        sample = np.zeros(self.nDims, dtype=np.float_)

        cdef int i
        for i in range(0, self.split):
            sample[i] = self.samplers[i].sample(unit_cube[i])

        for i in range(self.split, self.nDims):
            sample[i] = self.samplers[i].sample(unit_cube[i])

        return sample


    cpdef object _calculate_error_bars(self, np.ndarray [double, ndim=1] samples):
        """
        Return points at which the empirical cumulative distribution function (eCDF) cuts specified 
        percentage values.
        The returned values are used as error bars.
        """

        cdef double alpha = 0.68
        cdef double lower_percentage = (1 - alpha) / 2
        cdef double upper_percentage = lower_percentage + alpha

        cdef np.ndarray [double, ndim=1] x = np.sort(samples)
        cdef np.ndarray [double, ndim=1] cdf = np.arange(1, len(x) + 1) / np.float_(len(x))

        cdef int lower_index = np.argmin(np.abs(cdf - lower_percentage))
        cdef int upper_index = np.argmin(np.abs(cdf - upper_percentage))

        return x[lower_index], x[upper_index]



    cpdef object robustness_analysis(self, np.ndarray [long, ndim=1] optimal_n, np.ndarray [double, ndim=1] optimal_d):
        cdef int num_samples = 1000
        cdef int num_wavelengths = 1000
        cdef object rng = np.random.default_rng()
        cdef np.ndarray [double, ndim=2] samples = rng.normal(loc=optimal_d, scale=np.full(shape=len(optimal_d),
                                                        fill_value=1e-9), size=(num_samples, len(optimal_d)))
        cdef np.ndarray [complex, ndim=2] amplitude_s_samples = np.zeros((num_samples, num_wavelengths), dtype=complex)
        cdef np.ndarray [complex, ndim=2] amplitude_p_samples = np.zeros((num_samples, num_wavelengths), dtype=complex)

        cdef PrecomputedValues precomputed_values = self.precomputed_values_for_plotting
        cdef np.ndarray [complex, ndim=1] amplitude_s = np.zeros(precomputed_values.num_wavelengths, dtype=complex)
        cdef np.ndarray [complex, ndim=1] amplitude_p = np.zeros(precomputed_values.num_wavelengths, dtype=complex)

        # Calculate values at optimal_d.
        self.amplitude_wrapper(precomputed_values, optimal_n, optimal_d, amplitude_s, amplitude_p)
        cdef np.ndarray [double, ndim=2] values = np.zeros((5, num_wavelengths), dtype=np.float_)
        values[0, :] = np.abs(amplitude_s) ** 2
        values[1, :] = np.abs(amplitude_p) ** 2
        values[2, :] = np.abs(values[0] + values[1])
        values[3, :] = np.abs(values[0] - values[1])
        values[4, :] = np.angle(amplitude_s / amplitude_p)

        # Map samples to amplitudes.
        cdef int i
        cdef np.ndarray [double, ndim=1] sample
        for i in range(num_samples):
            sample = samples[i, :]
            self.amplitude_wrapper(precomputed_values, optimal_n, sample, amplitude_s, amplitude_p)
            amplitude_s_samples[i, :] = amplitude_s
            amplitude_p_samples[i, :] = amplitude_p

        # Map samples to reflectivities.
        cdef np.ndarray [object, ndim=1] all_samples = np.empty(5, dtype=object)
        # reflectivity_s
        all_samples[0] = np.abs(amplitude_s_samples) ** 2
        # reflectivity_p
        all_samples[1] = np.abs(amplitude_p_samples) ** 2
        # reflectivity_sum
        all_samples[2] = np.abs(all_samples[0] + all_samples[1])
        # reflectivity_diff
        all_samples[3] = np.abs(all_samples[0] - all_samples[1])
        # phase_difference
        all_samples[4] = np.angle(amplitude_s_samples / amplitude_p_samples)

        # First index: select (0) reflectivity_s (1) reflectivity_p (2) reflectivity_sum (3) reflectivity_diff
        # (4) phase_difference. Second index: select (0) values at optimal_d (1) lower (2) upper.
        cdef np.ndarray [double, ndim=3] results = np.zeros((5, 3, num_wavelengths), dtype=np.float_)
        cdef int j
        for i in range(5):
            results[i, 0, :] = values[i, :]

            samples = all_samples[i]
            for j in range(num_wavelengths):
                lower, upper = self._calculate_error_bars(samples[:, j])
                results[i, 1, j] = lower
                results[i, 2, j] = upper

        return np.asarray(precomputed_values.wavelengths), results