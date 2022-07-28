import numpy as np
from scipy.linalg import solve_banded


def reflectivity_s(M: np.int_, n: np.ndarray, d: np.ndarray, wavelength: np.float_, n_outer: np.float_,
                 n_substrate: np.float_, theta_outer: np.float_) -> np.float_:
    mat: np.ndarray = _make_matrix(M, n, d, 2 * np.pi / wavelength, n_outer, n_substrate, theta_outer)
    c: np.ndarray = _make_vector(M)
    x: np.ndarray = solve_banded(l_and_u=(2,2), ab=mat, b=c)
    return np.abs(x[0]) ** 2


def _make_matrix(M: np.int_, n: np.ndarray, d: np.ndarray, k_outer: np.float_, n_outer: np.float_, n_substrate: np.float_,
                 theta_outer: np.float_) -> np.ndarray:
    assert len(n) == M and len(d) == M

    def calc_k_x(n):
        return k_outer * np.sqrt((n / n_outer) ** 2 - np.sin(theta_outer) ** 2)

    k_x: np.ndarray = calc_k_x(n)
    k_x_substrate: np.float = calc_k_x(n_substrate)
    phi: np.ndarray = k_x * d

    alphas: list[np.ndarray] = [
        np.array([[-np.exp(1j * phi[j - 1]), -1], [-np.exp(1j * phi[j - 1]) * k_x[j - 1], k_x[j - 1]]])
        for j in range(1, M + 1)
    ]

    betas: list[np.ndarray] = [
        np.array([[1, np.exp(1j * phi[j])], [k_x[j], -k_x[j] * np.exp(1j * phi[j])]])
        for j in range(0, M)
    ]

    columns: list[np.ndarray] = [
        np.array([
            [0, betas[i][0, 1]],
            [betas[i][0, 0], betas[i][1, 1]],
            [betas[i][1, 0], alphas[i][0, 1]],
            [alphas[i][0, 0], alphas[i][1, 1]],
            [alphas[i][1, 0], 0]
        ])
        for i in range(0, M)
    ]

    mat: np.ndarray = np.hstack([
        np.array([[0.], [0.], [-1.], [1.], [0.]]),
        *columns,
        np.array([[0.], [1.], [k_x_substrate], [0.], [0.]])
    ])

    assert mat.shape[0] == 5 and mat.shape[1] == 2 * M + 2
    return mat

def _make_vector(M: np.int_) -> np.ndarray:
    c: np.ndarray = np.zeros(2 * M + 2)
    c[0] = 1
    c[1] = 1
    return c


def main():
    return


if __name__ == '__main__':
    main()
