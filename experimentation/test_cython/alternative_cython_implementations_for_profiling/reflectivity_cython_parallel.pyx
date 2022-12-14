# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Deactivate bounds checking, Deactivate negative indexing, No division by zero checking

import scipy.linalg
cimport numpy as np
import numpy as np
import cython
from cython.parallel import prange

cdef extern from "<complex.h>" nogil:
    double complex cexp(double complex z)
    double real(double complex z)
    double complex csqrt(double complex z)


cdef extern from "<math.h>" nogil:
    double cos(double arg)
    double sin(double arg)


cdef double PI = 3.14159265358979323846


cpdef double reflectivity(int polarisation, int M, np.ndarray [double, ndim=1] n, np.ndarray [double, ndim=1] d, double wavelength, double n_outer,
                 double n_substrate, double theta_outer):
    """
    Calculates the reflectivity.
    Function arguments are the same as for 'ampltiude()'.

    @return: Reflectivity |amplitude|^2
    """
    return np.abs(amplitude(polarisation,  M, n, d, wavelength, n_outer, n_substrate, theta_outer)) ** 2


cpdef double phase(int polarisation, int M, np.ndarray [double, ndim=1] n, np.ndarray [double, ndim=1] d, double wavelength, double n_outer,
                 double n_substrate, double theta_outer):
    """
    Calculates phase of the reflected complex amplitude.
    Function arguments are the same as for 'ampltiude()'.

    @return: Phase in the range (-pi, pi]
    """
    return np.angle(amplitude(polarisation,  M, n, d, wavelength, n_outer, n_substrate, theta_outer))


cpdef complex amplitude(int polarisation, int M, np.ndarray [double, ndim=1] n, np.ndarray [double, ndim=1] d, double wavelength, double n_outer,
                 double n_substrate, double theta_outer):
    """
    Calculates the reflected complex amplitude by solving the linear system Mx=b for x.
    Function arguments are the same as for '_make_matrix()'.
    """

    cdef np.ndarray [complex, ndim=2] mat = _make_matrix(polarisation, M, n, d, 2 * PI / wavelength, n_outer, n_substrate, theta_outer)
    cdef np.ndarray [complex, ndim=1] c = _make_vector(M)
    # M is a banded matrix so that 'solve_banded' can be used. Flags are set for performance.
    cdef np.ndarray [complex, ndim=1] x = scipy.linalg.solve_banded(l_and_u=(2, 2), ab=mat, b=c, overwrite_ab=True, overwrite_b=True, check_finite=False)
    return x[0]


cpdef np.ndarray [complex, ndim=2] _make_matrix(int polarisation, int M, np.ndarray [double, ndim=1] n, np.ndarray [double, ndim=1] d, double k_outer, double n_outer, double n_substrate, double theta_outer):
    cdef complex cos_theta_outer = cos(theta_outer)
    cdef np.ndarray [complex, ndim=1] cos_theta = np.sqrt(np.complex_(1 - (n_outer / n) ** 2 * np.sin(theta_outer) ** 2))
    cdef complex cos_theta_substrate = csqrt(1 - (n_outer / n_substrate) ** 2 * sin(theta_outer) ** 2)

    cdef np.ndarray [complex, ndim=1] exps = np.exp(1j * (k_outer * d) * (n / n_outer) * cos_theta)

    cdef complex a_mat = -1
    cdef complex b_mat = 1
    cdef complex c_mat = 1
    cdef complex d_mat = n_substrate * cos_theta_substrate

    cdef complex [:, :] mat = np.zeros((5, 2*M+2), dtype=complex)
    mat[2, 0] = a_mat
    mat[3, 0] = b_mat
    mat[1, 2 * M + 1] = c_mat
    mat[2, 2 * M + 1] = d_mat

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

    cdef int i
    for i in prange(1, M, nogil=True):
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

    return np.asarray(mat)


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