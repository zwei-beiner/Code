from typing import Union

import numpy as np
import numpy.typing as npt

import scipy.linalg


def reflectivity_s(M: np.int_, n: np.ndarray, d: np.ndarray, wavelength: np.float_, n_outer: np.float_,
                 n_substrate: np.float_, theta_outer: np.float_) -> np.float_:
    mat: npt.NDArray[np.complex_] = _make_matrix(M, n, d, 2 * np.pi / wavelength, n_outer, n_substrate, theta_outer)
    c: npt.NDArray[np.complex_] = _make_vector(M)
    x: npt.NDArray[np.complex_] = scipy.linalg.solve_banded(l_and_u=(2,2), ab=mat, b=c)
    return np.abs(x[0]) ** 2


def _make_matrix(M: np.int_, n: npt.NDArray[np.float_], d: npt.NDArray[np.float_], k_outer: np.float_, n_outer: np.float_, n_substrate: np.float_,
                 theta_outer: np.float_) -> npt.NDArray[np.complex_]:
    # assert len(n) == M and len(d) == M

    def calc_k_x(n: Union[np.float_, npt.NDArray[np.float_]]) -> Union[np.complex_, npt.NDArray[np.complex_]]:
        # Use np.emath.sqrt (instead of np.sqrt) to return complex numbers if argument of sqrt is negative
        return np.complex_(k_outer * np.emath.sqrt((n / n_outer) ** 2 - np.sin(theta_outer) ** 2))

    k_x_outer: np.complex_ = np.complex_(k_outer * np.cos(theta_outer))
    k_x: npt.NDArray[np.complex_] = calc_k_x(n)
    # return (n / n_outer) ** 2
    k_x_substrate: np.complex_ = calc_k_x(n_substrate)
    phi: npt.NDArray[np.complex_] = k_x * np.complex_(d)

    alphas: list[npt.NDArray[np.complex_]] = [
        np.array([[-np.exp(1j * phi[j - 1]), -1], [-np.exp(1j * phi[j - 1]) * k_x[j - 1], k_x[j - 1]]], dtype=np.complex_)
        for j in range(1, M + 1)
    ]

    betas: list[npt.NDArray[np.complex_]] = [
        np.array([[1, np.exp(1j * phi[0])],[k_x[0] / k_x_outer, -k_x[0] / k_x_outer * np.exp(1j * phi[0])]], dtype=np.complex_)
    ] + [
        np.array([[1, np.exp(1j * phi[j])], [k_x[j], -k_x[j] * np.exp(1j * phi[j])]], dtype=np.complex_)
        for j in range(1, M)
    ]

    columns: list[npt.NDArray[np.complex_]] = [
        np.array([
            [0, betas[i][0, 1]],
            [betas[i][0, 0], betas[i][1, 1]],
            [betas[i][1, 0], alphas[i][0, 1]],
            [alphas[i][0, 0], alphas[i][1, 1]],
            [alphas[i][1, 0], 0]
        ], dtype=np.complex_)
        for i in range(0, M)
    ]

    mat: npt.NDArray[np.complex_] = np.hstack([
        np.array([[0], [0], [-1], [1], [0]], dtype=np.complex_),
        *columns,
        np.array([[0], [1], [k_x_substrate], [0], [0]], dtype=np.complex_)
    ])

    # assert mat.shape[0] == 5 and mat.shape[1] == 2 * M + 2
    return mat

def _make_vector(M: np.int_) -> npt.NDArray[np.complex_]:
    c: np.ndarray = np.zeros(2 * M + 2, dtype=np.complex_)
    c[0] = 1
    c[1] = 1
    return c


def main():
    return


if __name__ == '__main__':
    main()
