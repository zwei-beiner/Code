cimport numpy as np
import numpy as np

from reflectivity_for_import cimport reflectivity_namespace
from Samplers_for_import cimport CategoricalSampler, UniformSampler
from PrecomputedValues_for_import cimport PrecomputedValues


cdef class BackendCalculations:
    """
    Class which handles the heavy numerical computations.
    """

    # Number of wavelengths. Note: The C type 'long' corresponds to the Python type 'int'.
    cdef long M
    # Incident angle.
    cdef double theta_outer

    # Array of type PrecomputedValues. Store automatically calculated wavelengths.
    cdef object[:] precomputed_values_array
    # Store wavelengths for plotting.
    cdef object precomputed_values_for_plotting
    #  Store wavelengths manually specified by user.
    cdef object precomputed_values_manually_specified

    # Variable used for determining how many sets of wavelength to pre-compute.
    # It is essentially the change in total thickness before a new set of wavelengths must be used.
    cdef double delta_D
    # Number of sets of automatically precomputed wavelengths.
    cdef int K_max

    # Fixed and unfixed indices and values of the layer specifications for the thicknesses and refractive indices.
    cdef long[:] d_fixed_indices
    cdef long[:] d_unfixed_indices
    cdef double[:] d_fixed_values
    cdef long[:] n_fixed_indices
    cdef long[:] n_unfixed_indices

    # Number of unfixed refractive indices. In the report, this is the integer 'p'. In the Optimiser class, this is
    # the variable 'self._split'.
    cdef long split
    # Number of dimensions of the optimisation problem. In the report, this is the integer 'p+q'. In the Optimiser
    # class, this is the variable 'self._nDims'.
    cdef long nDims
    # Sum of the thicknesses of the layers with fixed thicknesses. Since this never changes for the entirety of the
    # optimisation, it is precomputed and stored for efficiency.
    cdef double d_fixed_sum

    # Numpy array of Python objects which are Samplers.
    cdef object[:] samplers

    # Int array which acts as a boolean array and stores whether weighting factors are zero.
    # Used in merit function calculation.
    # If entry is 1, interpret as True. If entry is 0, interpret as false.
    # Note: An int array must be used because boolean memoryviews are not supported in Cython.
    # (See https://stackoverflow.com/questions/49058191/boolean-numpy-arrays-with-cython)
    cdef int[:] is_term_switched_on

    cdef object r

    def __init__(self,
                 long M,
                 object n_outer,
                 object n_substrate,
                 double theta_outer,
                 double D_max,
                 double min_wavelength,
                 double max_wavelength,
                 object layer_specification,
                 object merit_function_specification,
                 long[:] d_fixed_indices,
                 long[:] d_unfixed_indices,
                 double[:] d_fixed_values,
                 object d_unfixed_values,
                 long[:] n_fixed_indices,
                 long[:] n_unfixed_indices,
                 object n_unfixed_values,
                 long split,
                 long nDims,
                 custom_wavelengths=None):
        """

        @param M: Number of layers
        @param theta_outer: Incident angle
        @param D_max: Maximum total thickness of the multilayer stack
        @param min_wavelength: minimum specified wavelength
        @param max_wavelength: maximum specified wavelength
        @param layer_specification: Type is tuple[list[RefractiveIndex]]
        @param merit_function_specification: Type is MeritFunctionSpecification
        @param d_fixed_indices: Layers whose thicknesses are fixed
        @param d_unfixed_indices: Layers whose thicknesses are unfixed
        @param d_fixed_values: Thicknesses which are fixed
        @param d_unfixed_values: Type is list[tuple[float, float]]
        @param n_fixed_indices: Layers whose refractive indices are fixed
        @param n_unfixed_indices: Layers whose refractive indices are unfixed
        @param n_unfixed_values: Type is list[list[RefractiveIndex]]. For each layer whose refractive indices are
                        unfixed, store a list of refractive indices from which the optimiser must choose.
        @param split: Number of unfixed refractive indices.
        @param custom_wavelengths: Manually specified wavelengths by the user
        """

        # Create this class to call functions from the file 'reflectivity.pyx' because Cython does not support static
        # methods.
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

        # Store this value because it is used multiple times in the following calculations.
        cdef double const = 8. * (1. / min_wavelength - 1. / max_wavelength)
        # K_max is chosen such that all allowed values of K that can be computed are in [0,...,K_max]
        # (0 and K_max inclusive).
        self.K_max = np.floor(D_max * const) + 1

        # Fill up precomputed_values_array.
        self.precomputed_values_array = np.empty((self.K_max, ), dtype=object)
        self.delta_D = 1. / const
        cdef int i
        cdef double[:] wavelengths
        for i in range(self.K_max):
            # Introduce offset 1.5 in calculation of D to get a non-zero D at i=0.
            wavelengths = self.r.calculate_wavelengths(min_wavelength, max_wavelength, (i + 1.5) * self.delta_D)
            self.precomputed_values_array[i] = PrecomputedValues(self.M, n_outer, n_substrate, wavelengths,
                                                                 layer_specification, merit_function_specification)

        self.precomputed_values_for_plotting = PrecomputedValues(self.M, n_outer, n_substrate,
                                                        np.linspace(min_wavelength, max_wavelength, num=1000),
                                                        layer_specification, merit_function_specification)
        if custom_wavelengths is not None:
            self.precomputed_values_manually_specified = PrecomputedValues(self.M, n_outer, n_substrate,
                                                                custom_wavelengths, layer_specification,
                                                                merit_function_specification)

        # Precompute this value for efficiency.
        self.d_fixed_sum = np.sum(d_fixed_values)

        # Pre-allocate samplers.
        self.samplers = np.empty(self.nDims, dtype=object)
        for i in range(self.nDims):
            if i < self.split:
                self.samplers[i] = CategoricalSampler(len(n_unfixed_values[i]))
            else:
                self.samplers[i] = UniformSampler(d_unfixed_values[i - self.split][0], d_unfixed_values[i - self.split][1])

        # Precompute these values for efficiency.
        self.is_term_switched_on = np.zeros(5, dtype=np.int32)
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
        """
        Fills up the arrays 'amplitude_s' and 'amplitude_p' with amplitude calculation results.
        The amplitude calculation is determined by the 'precomputed_values' object.
                
        @param precomputed_values: Object which contains the wavelengths and other data for the calculation
        @param n_params: The refractive index part of the parameter vector
        @param d_params: The thicknesses part of the parameter vector
        @param amplitude_s: Empty preallocated array
        @param amplitude_p: Empty preallocated array
        @return: None
        """

        cdef int i, j
        cdef double wavelength
        cdef double n_outer
        cdef double n_substrate
        cdef int fixed_index
        cdef int unfixed_index

        # Pre-allocate arrays with the appropriate size.
        n = np.zeros(self.M, dtype=np.float_)
        d = np.zeros(self.M, dtype=np.float_)
        # Already fill indices where d is fixed. Note: This is not done in the __init__ method because we want to avoid
        # race conditions by pre-allocating an array which is written to in each merit function evaluation.
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
        """
        Evaluates the merit function at the parameter vector 'params'.
        Wavelengths are chosen automatically.
        """

        # Split the 'params' array into the refractive index part and the thicknesses part.
        cdef np.ndarray [long, ndim=1] n_params = np.int_(params[:self.split])
        cdef np.ndarray [double, ndim=1] d_params = params[self.split:]

        # Calculate the total thickness of the layer.
        cdef double total_thickness = d_params.sum() + self.d_fixed_sum
        # Decide which set of wavelengths to use.
        cdef int K = np.floor(total_thickness / self.delta_D)

        # Select the correct precomputed values.
        cdef PrecomputedValues p = self.precomputed_values_array[K]
        cdef int num_wavelengths = len(p.wavelengths)
        # Pre-allocate the arrays which store the complex amplitudes at each wavelength.
        cdef np.ndarray [complex, ndim=1] amplitude_s = np.zeros(num_wavelengths, dtype=complex)
        cdef np.ndarray [complex, ndim=1] amplitude_p = np.zeros(num_wavelengths, dtype=complex)

        # Fill amplitude_s and amplitde_p arrays.
        self.amplitude_wrapper(p, n_params, d_params, amplitude_s, amplitude_p)

        # Calculate the merit function value. Any terms for which the 'weighting' is zero are skipped by the if/else
        # clauses for efficiency.
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
            relative_phases = np.unwrap(np.angle(amplitude_s / amplitude_p))
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
            relative_phases = np.unwrap(np.angle(amplitude_s / amplitude_p))
            res = res + p.phase_weighting * np.mean(
                ((relative_phases - p.target_relative_phase) / p.weight_function_phase) ** 2)

        return res


    cpdef np.ndarray [double, ndim=1] prior(self, np.ndarray [double, ndim=1] unit_cube):
        """
        Map the unit hypercube [0, 1]^(self.nDims) to a parameter vector.
        @param unit_cube: A uniformly randomly sampled value from [0, 1]^(self.nDims)
        @return: The parameter vector of type numpy.ndarray
        """

        sample = np.zeros(self.nDims, dtype=np.float_)

        cdef int i
        # Refractive indices
        for i in range(0, self.split):
            sample[i] = self.samplers[i].sample(unit_cube[i])

        # Thicknesses
        for i in range(self.split, self.nDims):
            sample[i] = self.samplers[i].sample(unit_cube[i])

        return sample


    cpdef object _calculate_error_bars(self, np.ndarray [double, ndim=1] samples):
        """
        Return points at which the empirical cumulative distribution function (eCDF) cuts specified 
        percentage values.
        The returned values are used as error bars.
        This is the 68% central credibility interval.
        
        @param samples: Samples for which the error bars should be calculated
        """

        cdef double alpha = 0.68
        cdef double lower_percentage = (1 - alpha) / 2
        cdef double upper_percentage = lower_percentage + alpha

        # See the definition of the eCDF, e.g. on Wikipedia. If we plot the arrays cdf vs. x, we would see the eCDF.
        cdef np.ndarray [double, ndim=1] x = np.sort(samples)
        cdef np.ndarray [double, ndim=1] cdf = np.arange(1, len(x) + 1) / np.float_(len(x))

        cdef int lower_index = np.argmin(np.abs(cdf - lower_percentage))
        cdef int upper_index = np.argmin(np.abs(cdf - upper_percentage))

        return x[lower_index], x[upper_index]



    cpdef object robustness_analysis(self, np.ndarray [long, ndim=1] optimal_n, np.ndarray [double, ndim=1] optimal_d):
        """
        Calculate error bars and values at optimal_d (not the sample mean) for 
        (0) reflectivity_s
        (1) reflectivity_p
        (2) reflectivity_sum
        (3) reflectivity_diff
        (4) phase difference
        
        Samples are taken from a Gaussian with standard deviation 1nm in all directions, centred at the optimal solution
        optimal_d. For each sample, the functions (0)-(4) are evaluated and error bars are calculated.
        
        @param optimal_n: Result of the optimisation for the refractive indices
        @param optimal_d: Result of the optimisation for the thicknesses
        @return: Wavelengths array to be used as the x-axis for plotting, results which is a 3-dimensional numpy array
                containing the results of the robustness analysis.
        """

        # Choose to do the computation for 10^3 samples.
        cdef int num_samples = 1000
        # Choose 10^3 equally spaced wavelengths for high-resolution plotting.
        cdef int num_wavelengths = 1000
        # Initialise RNG.
        cdef object rng = np.random.default_rng()
        # Create a 2D array of Gaussian samples with standard deviation 1nm. First index: which sample? Second index:
        # which unfixed thickness?
        cdef np.ndarray [double, ndim=2] samples = rng.normal(loc=optimal_d, scale=np.full(shape=len(optimal_d),
                                                        fill_value=1e-9), size=(num_samples, len(optimal_d)))
        # Create 2D arrays which store the samples mapped to amplitudes.
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
        values[4, :] = np.unwrap(np.angle(amplitude_s / amplitude_p))

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
        # Important: The 'unwrap' function has to be provided an array because it compares neighbourinf values in the
        # array to decide whether it should undo a discontinuity. A way to think about this is that we have sampled a
        # set of functions (phase difference vs wavelength), and we are unwrapping each sampled function.
        all_samples[4] = np.unwrap(np.angle(amplitude_s_samples / amplitude_p_samples), axis=1)

        # First index: select (0) reflectivity_s (1) reflectivity_p (2) reflectivity_sum (3) reflectivity_diff
        # (4) phase_difference. Second index: select (0) values at optimal_d (1) lower (2) upper. Third index: which
        # wavelength?
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


    cpdef np.ndarray [double, ndim=2] calculate_critical_thicknesses(self, np.ndarray [long, ndim=1] optimal_n):
        """
        Given the optimal solution found by the optimisation, this function returns the critical thicknesses for each 
        layer as a function of wavelength. 
        d_crit is chosen such that the transfer matrix of the layer is close to the identity.
        
        @param optimal_n: Optimal refractive indices found by the optimisation
        @return: 2D numpy array where the 1st index is the layer and the 2nd index is the wavelength
        """

        # 1000 wavelengths for high-resolution plotting.
        cdef PrecomputedValues p = self.precomputed_values_for_plotting
        cdef np.ndarray [double, ndim=1] wavelengths = np.asarray(p.wavelengths)
        cdef np.ndarray [double, ndim=2] d_crit = np.zeros((self.M, len(wavelengths)), dtype=np.float_)
        cdef np.ndarray [double, ndim=1] n_outer = np.asarray(p.n_outer)

        cdef int i
        cdef np.ndarray [double, ndim=1] n
        cdef np.ndarray [complex, ndim=1] cos_theta
        cdef np.ndarray [complex, ndim=1] k_x
        for i in range(self.M):
            n = np.float64(p.ns[i][optimal_n[i]])
            cos_theta = np.sqrt(np.complex_(1 - (n_outer / n) ** 2 * np.sin(self.theta_outer) ** 2))
            k_x = 2 * np.pi / wavelengths * n * cos_theta
            d_crit[i, :] = np.abs(1 / k_x)

        return 0.01 * d_crit