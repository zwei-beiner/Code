import scipy.linalg
cimport numpy as np
import numpy as np


cdef extern from "<complex.h>" nogil:
    double complex cexp(double complex z)
    double real(double complex z)
    double complex csqrt(double complex z)


cdef extern from "<math.h>" nogil:
    double cos(double arg)
    double sin(double arg)


cdef double PI = 3.14159265358979323846

cdef class reflectivity_namespace:
    """
    To 'cimport' Cython functions in another .pyx file, the functions in this file have to be wrapped in a class.
    Otherwise, a '__pyx_capi__ not defined' error is thrown.
    See https://stackoverflow.com/questions/56560007/building-python-package-containing-multiple-cython-extensions/56807269#56807269
    """

    cpdef complex[:] amplitude_at_wavelengths(self, int polarisation, int M, np.ndarray [double, ndim=1] n, np.ndarray [double, ndim=1] d, np.ndarray [double, ndim=1] wavelengths, double n_outer, double n_substrate, double theta_outer):
        # Cython does not allow np.float_ inside the square brackets, only the type 'double'. But Cython does not allow
        # dtype=double. Hence, we must use 'double' in the square brackets and 'dtype=np.float_'. This is not a problem
        # because 'np.float_' is an alias for the C-type 'double'.
        cdef np.ndarray [complex, ndim=1] amplitudes = np.zeros(n, dtype=complex)
        cdef int i
        for i in range(len(wavelengths)):
            amplitudes[i] = self.amplitude(polarisation, M, n, d, wavelengths[i], n_outer, n_substrate, theta_outer)

        return amplitudes


    cpdef double[:] calculate_wavelengths(self, double min_wavelength, double max_wavelength, double total_thickness):
        # cdef double total_thickness = d.sum()
        cdef double f = 1 / (8 * total_thickness)

        # Calculate number of points
        cdef double temp = min_wavelength
        cdef int n = 0
        while temp < max_wavelength:
            temp += f * temp ** 2
            n += 1

        # Store wavelengths
        cdef np.ndarray [double, ndim=1] wavelengths = np.zeros(n, dtype=np.float_)
        temp = min_wavelength
        cdef int i
        for i in range(n):
            wavelengths[i] = temp
            temp += f * temp ** 2

        cdef double[:] wave = wavelengths
        return wavelengths


    cpdef complex amplitude(self, int polarisation, int M, double[:] n, double[:] d, double wavelength, double n_outer,
                     double n_substrate, double theta_outer):
        """
        Calculates the reflected complex amplitude by solving the linear system Mx=b for x.
        Function arguments are the same as for '_make_matrix()'.
        """

        # Cast memoryviews (double[:]) into numpy arrays using np.asarray because otherwise a runtime error
        # "Cannot convert calculation.reflectivity._memoryviewslice to numpy.ndarray" is thrown.
        cdef np.ndarray [complex, ndim=2] mat = _make_matrix(polarisation, M, np.asarray(n), np.asarray(d), 2 * PI / wavelength, n_outer, n_substrate, theta_outer)
        cdef np.ndarray [complex, ndim=1] c = _make_vector(M)
        # M is a banded matrix so that 'solve_banded' can be used. Flags are set for performance.
        cdef np.ndarray [complex, ndim=1] x = scipy.linalg.solve_banded(l_and_u=(2, 2), ab=mat, b=c, overwrite_ab=True, overwrite_b=True, check_finite=False)
        return x[0]


cpdef np.ndarray [complex, ndim=2] _make_matrix(int polarisation, int M, np.ndarray [double, ndim=1] n, np.ndarray [double, ndim=1] d, double k_outer, double n_outer, double n_substrate, double theta_outer):
    """
    Constructs the matrix \mathbf{M} in band structure form.

    @param polarisation: 0 for s-polarisation and 1 for p-polarisation
    @param M: Number of layers. Valid input range: M ≥ 1
    @param n: Array storing the refractive indices of the layers.
    @param d: Array storing the thicknesses of the layers.
    @param k_outer: Wavenumber "2*pi/lambda" of the incident light in the outer medium. lambda is wavelength in the
    outer medium
    @param n_outer: Refractive index of the outer medium.
    @param n_substrate: Refractive index of the substrate.
    @param theta_outer: Angle in the outer medium, i.e. incident angle. Valid range: -pi/2 < theta_outer < pi/2
    @return: Matrix \mathbf{M}
    """

    cdef complex cos_theta_outer = cos(theta_outer)
    cdef np.ndarray [complex, ndim=1] cos_theta = np.sqrt(np.complex_(1 - (n_outer / n) ** 2 * np.sin(theta_outer) ** 2))
    cdef complex cos_theta_substrate = csqrt(1 - (n_outer / n_substrate) ** 2 * sin(theta_outer) ** 2)

    # Pre-compute complex exponentials for performance increase.
    cdef np.ndarray [complex, ndim=1] exps = np.exp(1j * (k_outer * d) * (n / n_outer) * cos_theta)

    cdef np.ndarray [complex, ndim=2] mat = np.zeros((5, 2*M+2), dtype=complex)

    cdef int i
    if polarisation == 0:
        mat[2, 0] = -1
        mat[3, 0] = 1
        mat[1, 2 * M + 1] = 1
        mat[2, 2 * M + 1] = n_substrate * cos_theta_substrate

        mat[0, 1] = 0
        mat[1, 1] = 1
        mat[2, 1] = (n[0] / n_outer) * (cos_theta[0] / cos_theta_outer)
        mat[3, 1] = -exps[0]
        mat[4, 1] = -exps[0] * n[0] * cos_theta[0]

        mat[0, 2] = exps[0]
        mat[1, 2] = -(n[0] / n_outer) * (cos_theta[0] / cos_theta_outer) * exps[0]
        mat[2, 2] = -1
        mat[3, 2] = n[0] * cos_theta[0]
        mat[4, 2] = 0

        for i in range(1, M):
            mat[0, 2 * i + 1] = 0
            mat[1, 2 * i + 1] = 1
            mat[2, 2 * i + 1] = n[i] * cos_theta[i]
            mat[3, 2 * i + 1] = -exps[i]
            mat[4, 2 * i + 1] = -exps[i] * n[i] * cos_theta[i]

            mat[0, 2 * i + 2] = exps[i]
            mat[1, 2 * i + 2] = -n[i] * cos_theta[i] * exps[i]
            mat[2, 2 * i + 2] = -1
            mat[3, 2 * i + 2] = n[i] * cos_theta[i]
            mat[4, 2 * i + 2] = 0

    elif polarisation == 1:
        mat[2, 0] = -1
        mat[3, 0] = 1
        mat[1, 2 * M + 1] = cos_theta_substrate
        mat[2, 2 * M + 1] = n_substrate

        mat[0, 1] = 0
        mat[1, 1] = cos_theta[0] / cos_theta_outer
        mat[2, 1] = n[0] / n_outer
        mat[3, 1] = -exps[0] * cos_theta[0]
        mat[4, 1] = -exps[0] * n[0]

        mat[0, 2] = exps[0] * cos_theta[0] / cos_theta_outer
        mat[1, 2] = -exps[0] * n[0] / n_outer
        mat[2, 2] = -cos_theta[0]
        mat[3, 2] = n[0]
        mat[4, 2] = 0

        for i in range(1, M):
            mat[0, 2 * i + 1] = 0
            mat[1, 2 * i + 1] = cos_theta[i]
            mat[2, 2 * i + 1] = n[i]
            mat[3, 2 * i + 1] = -exps[i] * cos_theta[i]
            mat[4, 2 * i + 1] = -exps[i] * n[i]

            mat[0, 2 * i + 2] = exps[i] * cos_theta[i]
            mat[1, 2 * i + 2] = -exps[i] * n[i]
            mat[2, 2 * i + 2] = -cos_theta[i]
            mat[3, 2 * i + 2] = n[i]
            mat[4, 2 * i + 2] = 0

    else:
        raise ValueError(f'Invalid argument: polarisation = {polarisation}')

    return mat


cpdef np.ndarray [complex, ndim=1] _make_vector(int M):
    """
    Constructs the vector \mathbf{c}.
    @param M: Number of layers
    @return: Vector \mathbf{c}
    """

    cdef np.ndarray [complex, ndim=1] c = np.zeros(2 * M + 2, dtype=complex)
    c[0] = 1
    c[1] = 1
    return c