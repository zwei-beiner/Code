from typing import Union

import numpy as np
import numpy.typing as npt

import scipy.linalg


def reflectivity(polarisation: int, M: np.int_, n: np.ndarray, d: np.ndarray, wavelength: np.float_, n_outer: np.float_,
                 n_substrate: np.float_, theta_outer: np.float_) -> np.float_:
    """
    Calculates the reflectivity.
    Function arguments are the same as for 'ampltiude()'.

    @return: Reflectivity |amplitude|^2
    """

    # Note: locals() returns a dictionary with all local variables of the function. When called here, it returns all the
    # function arguments.
    return np.abs(amplitude(**locals())) ** 2


def phase(polarisation: int, M: np.int_, n: np.ndarray, d: np.ndarray, wavelength: np.float_, n_outer: np.float_,
                 n_substrate: np.float_, theta_outer: np.float_) -> np.float_:
    """
    Calculates phase of the reflected complex amplitude.
    Function arguments are the same as for 'ampltiude()'.

    @return: Phase in the range (-pi, pi]
    """
    return np.float_(np.angle(amplitude(**locals())))

def amplitude(polarisation: int, M: np.int_, n: np.ndarray, d: np.ndarray, wavelength: np.float_, n_outer: np.float_,
                 n_substrate: np.float_, theta_outer: np.float_) -> np.complex_:
    """
    Calculates the reflected complex amplitude by solving the linear system Mx=b for x.
    Function arguments are the same as for '_make_matrix()'.
    """

    mat: npt.NDArray[np.complex_] = _make_matrix(polarisation, M, n, d, 2 * np.pi / wavelength, n_outer, n_substrate, theta_outer)
    c: npt.NDArray[np.complex_] = _make_vector(M)
    # M is a banded matrix so that 'solve_banded' can be used. Flags are set for performance.
    x: npt.NDArray[np.complex_] = scipy.linalg.solve_banded(l_and_u=(2, 2), ab=mat, b=c, overwrite_ab=True, overwrite_b=True, check_finite=False)
    return x[0]


def _make_matrix(polarisation: int, M: np.int_, n: npt.NDArray[np.float_], d: npt.NDArray[np.float_], k_outer: np.float_, n_outer: np.float_, n_substrate: np.float_,
                 theta_outer: np.float_) -> npt.NDArray[np.complex_]:
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
    # assert len(n) == M and len(d) == M

    # Concatenate arrays to call np.sqrt only once, which increases performance.
    all_n = np.concatenate([np.array([n_outer]), n, np.array([n_substrate])])
    all_cos_theta = np.sqrt(np.complex_(1 - (n_outer / all_n) ** 2 * np.sin(theta_outer) ** 2))

    cos_theta_outer: np.complex_ = all_cos_theta[0]
    cos_theta: npt.NDArray[np.complex_] = all_cos_theta[1:M+1]
    cos_theta_substrate: np.complex_ = all_cos_theta[M+1]

    # Pre-compute complex exponentials for performance increase.
    exps: npt.NDArray[np.complex_] = np.exp(1j * (k_outer * d) * (n / n_outer) * cos_theta)

    if polarisation == 0:
        (a, b, c, d, columns) = _make_s_pol_submatrices(M, exps, cos_theta, cos_theta_outer, cos_theta_substrate, n, n_outer, n_substrate)
    elif polarisation == 1:
        (a, b, c, d, columns) = _make_p_pol_submatrices(M, exps, cos_theta, cos_theta_outer, cos_theta_substrate, n, n_outer,
                                                              n_substrate)
    else:
        raise ValueError(f'Invalid argument: polarisation = {polarisation}')

    mat: npt.NDArray[np.complex_] = np.hstack([
        np.array([[0], [0], [a], [b], [0]], dtype=np.complex_),
        *columns,
        np.array([[0], [c], [d], [0], [0]], dtype=np.complex_)
    ])

    #assert mat.shape[0] == 5 and mat.shape[1] == 2 * M + 2
    return mat


def _make_s_pol_submatrices(M: np.int_, exps: npt.NDArray[np.complex_], cos_theta: npt.NDArray[np.complex_],
                            cos_theta_outer: np.complex_, cos_theta_substrate: np.complex_, n: npt.NDArray[np.float_],
                            n_outer: np.float_, n_substrate: np.float_) \
        -> tuple[np.complex_, np.complex_, np.complex_, np.complex_, list[npt.NDArray[np.complex_]]]:
    a: np.complex_ = np.complex_(-1)
    b: np.complex_ = np.complex_(1)
    c: np.complex_ = np.complex_(1)
    d: np.complex_ = np.complex_(n_substrate * cos_theta_substrate)

    columns: list[npt.NDArray[np.complex_]] = [
        np.array([
                [0, exps[0]],
                [1, -(n[0] / n_outer) * (cos_theta[0] / cos_theta_outer) * exps[0]],
                [(n[0] / n_outer) * (cos_theta[0] / cos_theta_outer), -1],
                [-exps[0], n[0] * cos_theta[0]],
                [-exps[0] * n[0] * cos_theta[0], 0]
            ], dtype=np.complex_)
        ] + [
            np.array([
                [0, exps[j]],
                [1, -n[j] * cos_theta[j] * exps[j]],
                [n[j] * cos_theta[j], -1],
                [-exps[j], n[j] * cos_theta[j]],
                [-exps[j] * n[j] * cos_theta[j], 0]
            ])
            for j in range(1, M)
        ]

    return (a, b, c, d, columns)


def _make_p_pol_submatrices(M: np.int_, exps: npt.NDArray[np.complex_], cos_theta: npt.NDArray[np.complex_],
                            cos_theta_outer: np.complex_, cos_theta_substrate: np.complex_, n: npt.NDArray[np.float_], n_outer: np.float_, n_substrate: np.float_) \
        -> tuple[np.complex_, np.complex_, np.complex_, np.complex_, list[npt.NDArray[np.complex_]]]:
    a: np.complex_ = np.complex_(-1)
    b: np.complex_ = np.complex_(1)
    c: np.complex_ = np.complex_(cos_theta_substrate)
    d: np.complex_ = np.complex_(n_substrate)

    columns: list[npt.NDArray[np.complex_]] = [
        np.array([
            [0,exps[0] * cos_theta[0] / cos_theta_outer],
            [cos_theta[0] / cos_theta_outer,-exps[0] * n[0] / n_outer],
            [n[0] / n_outer,-cos_theta[0]],
            [-exps[0] * cos_theta[0],n[0]],
            [-exps[0] * n[0], 0]
        ])
    ] + [
        np.array([
            [0, exps[j] * cos_theta[j]],
            [cos_theta[j], -exps[j] * n[j]],
            [n[j], -cos_theta[j]],
            [-exps[j] * cos_theta[j], n[j]],
            [-exps[j] * n[j], 0],
        ])
        for j in range(1, M)
    ]

    return (a, b, c, d, columns)

def _make_vector(M: np.int_) -> npt.NDArray[np.complex_]:
    """
    Constructs the vector \mathbf{c}.
    @param M: Number of layers
    @return: Vector \mathbf{c}
    """

    c: np.ndarray = np.zeros(2 * M + 2, dtype=np.complex_)
    c[0] = 1
    c[1] = 1
    return c


def main():
    return


if __name__ == '__main__':
    main()
