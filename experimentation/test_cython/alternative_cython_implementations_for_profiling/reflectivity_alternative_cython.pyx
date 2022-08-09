# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Deactivate bounds checking, Deactivate negative indexing, No division by zero checking

cimport numpy as np
import numpy as np
import cython

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
    cdef complex cos_theta_outer = cos(theta_outer)
    cdef np.ndarray [complex, ndim=1] cos_theta = np.sqrt(np.complex_(1 - (n_outer / n) ** 2 * np.sin(theta_outer) ** 2))
    cdef complex cos_theta_substrate = csqrt(1 - (n_outer / n_substrate) ** 2 * sin(theta_outer) ** 2)

    # Calculate transfer matrix T
    cdef complex[2][2] T = [[1, 0], [0, 1]]
    cdef complex[2][2] temp
    cdef complex[2][2] other_temp
    cdef complex phi

    T_ij(temp, n_outer, cos_theta_outer, n[0], cos_theta[0])
    matmul(T, temp, other_temp)

    cdef int i
    for i in range(1, M):
        phi = 2 * PI / wavelength * n[i - 1] / n_outer * d[i - 1] * cos_theta[i - 1]
        T_i(temp, phi)
        matmul(T, temp, other_temp)

        T_ij(temp, n[i - 1], cos_theta[i - 1], n[i], cos_theta[i])
        matmul(T, temp, other_temp)

    phi = 2 * PI / wavelength * n[M - 1] / n_outer * d[M - 1] * cos_theta[M - 1]
    T_i(temp, phi)
    matmul(T, temp, other_temp)

    T_ij(temp, n[M - 1], cos_theta[M - 1], n_substrate, cos_theta_substrate)
    matmul(T, temp, other_temp)

    return T[1][0] / T[0][0]

cdef void T_i(complex[:, :] temp, complex phi):
    temp[0][0] = cexp(-1j * phi)
    temp[0][1] = 0
    temp[1][0] = 0
    temp[1][1] = cexp(1j * phi)

cdef void T_ij(complex[:, :] temp, double n_i, complex cos_theta_i, double n_j, complex cos_theta_j):
    cdef complex r = r_ij_s(n_i, cos_theta_i, n_j, cos_theta_j)
    temp[0][0] = 1
    temp[0][1] = r
    temp[1][0] = r
    temp[1][1] = 1


cdef void matmul(complex[:, :] A, complex[:, :] B, complex[:, :] other_temp):
    """
    Calculates AB and stores result in A
    """
    other_temp[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0]
    other_temp[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1]
    other_temp[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    other_temp[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1]

    cdef int i
    cdef int j
    for i in range(2):
        for j in range(2):
            A[i, j] = other_temp[i][j]

cdef complex r_ij_s(double n_i, complex cos_theta_i, double n_j, complex cos_theta_j):
    return (<complex>n_i * cos_theta_i - <complex>n_j * cos_theta_j) / (<complex>n_i * cos_theta_i + <complex>n_j * cos_theta_j)
