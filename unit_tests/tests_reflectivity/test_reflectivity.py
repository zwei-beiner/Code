from pathlib import Path
from unittest import TestCase

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

from main import reflectivity, amplitude, _make_matrix


class Test_reflectivity(TestCase):
    # Note: Can't handle zero layers (M=0)
    # def test_fresnel_equations(self):
    #     def fresnel_reflection_coefficient_for_s_pol(n_1: np.float_, n_2: np.float_, theta_incident: np.float_):
    #         return np.abs((n_1 * np.cos(theta_incident) - n_2 * np.emath.sqrt(1 - (n_1 * np.sin(theta_incident) / n_2) ** 2))
    #                       / (n_1 * np.cos(theta_incident) + n_2 * np.emath.sqrt(1 - (n_1 * np.sin(theta_incident) / n_2) ** 2))) ** 2
    #
    #     M = 0
    #     for n_1 in np.linspace(0.1, 10, num=10):
    #         for n_2  in np.linspace(0.1, 10, num=10):
    #             for theta_incident in np.linspace(-np.pi/2 + 1e-6, np.pi - 1e-6, num=10):
    #                 self.assertAlmostEqual(
    #                     fresnel_reflection_coefficient_for_s_pol(n_1, n_2, theta_incident),
    #                     reflectivity_s(M, np.array([0]), np.array([0]), np.float_(500e-9), n_1, n_2, theta_incident),
    #                     delta=1e-15
    #                 )

    def test_comparison_with_transfer_matrices(self):
        def amplitude_from_transfer_matrix(polarisation: int, M: np.int_, n: np.ndarray, d: np.ndarray, wavelength: np.float_, n_outer: np.float_,
             n_substrate: np.float_, theta_outer: np.float_) -> npt.NDArray[np.complex_]:
            # Angles in layer and substrate
            def calc_cos_theta_i(n):
                # Snell's law
                sin_theta_i = n_outer * np.sin(theta_outer) / n
                # Note: Cosine can be complex
                cos_theta_i = np.emath.sqrt(1 - sin_theta_i ** 2)
                return cos_theta_i

            cos_theta_outer = np.cos(theta_outer)
            cos_theta_layer = calc_cos_theta_i(n)
            cos_theta_substrate = calc_cos_theta_i(n_substrate)

            n = np.append(n, n_substrate)
            cos_theta_layer = np.append(cos_theta_layer, cos_theta_substrate)

            # Fresnel coefficents for s polarisation (TE polarisation)
            def r_ij_s(n_i, cos_theta_i, n_j, cos_theta_j):
                return (n_i * cos_theta_i - n_j * cos_theta_j) / (n_i * cos_theta_i + n_j * cos_theta_j)

            def t_ij_s(n_i, cos_theta_i, n_j, cos_theta_j):
                return 2 * n_i * cos_theta_i / (n_i * cos_theta_i + n_j * cos_theta_j)

            # Fresnel coefficients for p polarisation (TM polarisation)
            def r_ij_p(n_i, cos_theta_i, n_j, cos_theta_j):
                return -(n_j * cos_theta_i - n_i * cos_theta_j) / (n_j * cos_theta_i + n_i * cos_theta_j)

            def t_ij_p(n_i, cos_theta_i, n_j, cos_theta_j):
                return 2 * n_i * cos_theta_i / (n_j * cos_theta_i + n_i * cos_theta_j)

            def transfer_matrix(r_ij, t_ij):
                def T_ij(n_i, cos_theta_i, n_j, cos_theta_j):
                    return np.array([
                        [1, r_ij(n_i, cos_theta_i, n_j, cos_theta_j)],
                        [r_ij(n_i, cos_theta_i, n_j, cos_theta_j), 1]
                    ]) / t_ij(n_i, cos_theta_i, n_j, cos_theta_j)

                def T_i(n_i, cos_theta_i, d_i):
                    phi = 2 * np.pi / wavelength * n_i / n_outer * d_i * cos_theta_i
                    return np.array([
                        [np.exp(-1j * phi), 0],
                        [0, np.exp(1j * phi)]
                    ])

                T = np.eye(2)
                T_01 = T_ij(n_outer, cos_theta_outer, n[0], cos_theta_layer[0])
                T = np.matmul(T, T_01)
                for i in range(1, M+1):
                    T = np.matmul(T, T_i(n[i - 1], cos_theta_layer[i - 1], d[i - 1]))
                    T = np.matmul(T, T_ij(n[i - 1], cos_theta_layer[i - 1], n[i], cos_theta_layer[i]))
                return T

            if polarisation == 0:
                T = transfer_matrix(r_ij_s, t_ij_s)
            else:
                T = transfer_matrix(r_ij_p, t_ij_p)


            return T[1, 0] / T[0, 0]

        rng = np.random.default_rng(0)
        num = 7
        for M in range(1, 10):
            print(M)
            for n in (rng.uniform(low=0.1, high=10, size=M) for _ in range(num)):
                for d in (rng.uniform(low=1, high=1e3, size=M) * 1e-9 for _ in range(num)):
                    for wavelength in np.linspace(200, 3000, num=num) * 1e-9:
                        for n_outer in np.linspace(0.1, 10, num=num):
                            for n_substrate in np.linspace(0.1, 10, num=num):
                                for theta_outer in np.linspace(-np.pi/2 + 1e-3, np.pi/2 - 1e-3, num=num):
                                    for polarisation in 0, 1:
                                        args = polarisation, M, n, d, wavelength, n_outer, n_substrate, theta_outer
                                        b_0_transfer_matrix = amplitude_from_transfer_matrix(*args)
                                        b_0_to_be_tested = amplitude(*args)
                                        self.assertAlmostEqual(b_0_transfer_matrix.real, b_0_to_be_tested.real, delta=2e-8)
                                        self.assertAlmostEqual(b_0_transfer_matrix.imag, b_0_to_be_tested.imag, delta=4e-10)


    def test_r_and_p_polarisation_give_the_same_amplitude_for_normal_incidence(self):
        rng = np.random.default_rng(0)

        theta_outer = 0

        num = 7
        for M in range(1, 10):
            for n in (rng.uniform(low=0.1,high=10,size=M) for _ in range(num)):
                for d in (rng.uniform(low=1,high=1e3,size=M) * 1e-9 for _ in range(num)):
                    for wavelength in np.linspace(200, 3000, num=num) * 1e-9:
                        for n_outer in np.linspace(0.1, 10, num=num):
                            for n_substrate in np.linspace(0.1, 10, num=num):
                                args = M, n, d, wavelength, n_outer, n_substrate, theta_outer
                                b_0_s = amplitude(0, *args)
                                b_0_p = amplitude(1, *args)

                                # Compare absolute value and phase
                                np.testing.assert_allclose(np.abs(b_0_s), np.abs(b_0_p), rtol=0, atol=7e-14)
                                np.testing.assert_allclose(np.angle(b_0_s), np.angle(b_0_p), rtol=0, atol=5e-13)


    def test_make_matrix(self):
        """
        Tests if matrix is constructed correctly for a single layer (which is a Fabry-Perot interferometer).
        """

        def make_test_matrix_for_fabry_pelot_etalon(n: np.float_, d: np.float_, k_outer: np.float_,
                                                    n_outer: np.float_, n_substrate: np.float_,
                                                    theta_outer: np.float_) -> tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_]]:
            cos_theta_outer = np.complex_(np.cos(theta_outer))
            cos_theta_layer = np.complex_(np.emath.sqrt(1 - (n_outer / n) ** 2 * np.sin(theta_outer) ** 2))
            cos_theta_substrate = np.complex_(np.emath.sqrt(1 - (n_outer / n_substrate) ** 2 * np.sin(theta_outer) ** 2))

            phi_0 = np.complex_((k_outer * d) * (n / n_outer) * cos_theta_layer)

            # Matrix for s polarisation
            mat_s: npt.NDArray[np.complex_] = np.array([
                [0,     0,                                                      np.exp(1j * phi_0),                                                           0],
                [0,     1,                                                      -(n / n_outer) * (cos_theta_layer / cos_theta_outer) * np.exp(1j * phi_0),    1],
                [-1,    (n / n_outer) * (cos_theta_layer / cos_theta_outer),    -1,                                                                           n_substrate * cos_theta_substrate],
                [1,     -np.exp(1j * phi_0),                                    n * cos_theta_layer,                                                          0],
                [0,     -np.exp(1j * phi_0) * n * cos_theta_layer,              0,                                                                            0]
            ], dtype=np.complex_)

            # Matrix for p polarisation
            mat_p: npt.NDArray[np.complex_] = np.array([
                [0,     0,                                      np.exp(1j * phi_0) * cos_theta_layer / cos_theta_outer,   0],
                [0,     cos_theta_layer / cos_theta_outer,    -np.exp(1j * phi_0) * n / n_outer,                          cos_theta_substrate],
                [-1,    n / n_outer,                            -cos_theta_layer,                                         n_substrate],
                [1,     -np.exp(1j * phi_0) * cos_theta_layer,    n,                                                      0],
                [0,     -np.exp(1j * phi_0) * n,                0,                                                        0]
            ], dtype=np.complex_)

            return mat_s, mat_p

        num = 7
        for n in np.linspace(0.1, 10, num=num):
            for d in np.linspace(1, 1e3, num=num) * 1e-9:
                for wavelength in np.linspace(200, 3000, num=num) * 1e-9:
                    print(f'Testing: n = {n}, wavelength = {wavelength}')
                    for n_outer in np.linspace(0.1, 10, num=num):
                        for n_substrate in np.linspace(0.1, 10, num=num):
                            for theta in np.linspace(0, np.pi / 2 - 1e-3, num=num):
                                R_s_correct: npt.NDArray[np.complex_]
                                R_p_correct: npt.NDArray[np.complex_]
                                R_s_correct, R_p_correct = make_test_matrix_for_fabry_pelot_etalon(
                                    n, d, 2 * np.pi / wavelength, n_outer, n_substrate, theta
                                )

                                M = 1
                                args = M, np.array([n]), np.array([d]), 2 * np.pi / wavelength, n_outer, n_substrate, theta
                                R_s_to_be_tested: npt.NDArray[np.complex_] = _make_matrix(0, *args)
                                R_p_to_be_tested: npt.NDArray[np.complex_] = _make_matrix(1, *args)

                                for polarisation, R_correct, R_to_be_tested in ('s', R_s_correct, R_s_to_be_tested), ('p', R_p_correct, R_p_to_be_tested):
                                    err_msg: str = f'polarisation: {polarisation}, n: {n}, d: {d}, wavelength: {wavelength}, n_outer: {n_outer}, ' \
                                                   f'n_substrate: {n_substrate}, theta: {theta}' + \
                                                   '\nR_correct:'+str(R_correct)+ \
                                                   '\n R_to_be_tested:'+str(R_to_be_tested)+ \
                                                   '\n Difference:'+str(R_correct - R_to_be_tested)
                                    np.testing.assert_allclose(R_correct, R_to_be_tested, rtol=0, atol=1e-15, err_msg=err_msg)


    def test_fabry_pelot_etalon(self):
        def calculate_amplitudes_using_analytic_formulas(wavelength: np.float_, d: np.float_, n_outer: np.float_, n_layer: np.float_, n_substrate: np.float_, theta_outer: np.float_) -> tuple[np.complex_, np.complex_]:
            """
            Calculates the reflected amplitude directly.

            @param wavelength: wavelength in the outer medium. Hence, wavelength in vacuum is "wavelength/n_outer"
            @param d: thickness of the layer in meters
            @param n_outer: refractive index of the outer medium
            @param n_layer: refractive index of the inner medium
            @param n_substrate: refractive index of the outer medium. Must be the same as the n_outer
            @param theta_outer: angle of incidence, i.e. angle to the normal in the outer medium. Can be in the range -pi/2<theta_outer<pi/2
            """

            costhetalayer = np.emath.sqrt(1 - (n_outer * np.sin(theta_outer) / n_layer) ** 2)

            phi_0 = (2 * np.pi / wavelength * d) * (n_layer / n_outer) * costhetalayer
            cos_theta_outer = np.cos(theta_outer)
            cos_theta_substrate = np.emath.sqrt(1 - (n_outer * np.sin(theta_outer) / n_substrate) ** 2)

            r_s = (n_outer * cos_theta_outer - n_layer * costhetalayer) / (n_outer * cos_theta_outer + n_layer * costhetalayer)
            r_p = -(n_layer * cos_theta_outer - n_outer * costhetalayer) / (n_layer * cos_theta_outer + n_outer * costhetalayer)

            def amplitude(r, phi_0):
                numerator = r * (1 - np.exp(2 * 1j * phi_0))
                denominator = (1 - (r ** 2) * np.exp(2 * 1j * phi_0))
                return numerator / denominator

            amplitude_s = amplitude(r_s, phi_0)
            amplitude_p = amplitude(r_p, phi_0)

            # print(f'r_s: {r_s}, phi_0: {phi_0}, '
            #       f'amplitude (problem): {r_s * (1 - np.exp(2 * 1j * phi_0)) / (1 - (r_s ** 2) * np.exp(2 * 1j * phi_0))}'
            #       f'amplitude (new): {numerator / denominator}'
            #       f'reflectivity: {np.abs(r_s * (1 - np.exp(2 * 1j * phi_0) / (1 - (r_s ** 2) * np.exp(2 * 1j * phi_0)))) ** 2}, '
            #       f'numerator: {(1 - np.exp(2 * 1j * phi_0))}, '
            #       f'denominator: {(1 - (r_s ** 2) * np.exp(2 * 1j * phi_0))}'
            #       f'new refkectivity: {np.abs(amplitude_s) ** 2}')
            return amplitude_s, amplitude_p

        M: int = 1

        for wavelength in np.linspace(200, 3000, num=10) * 1e-9:
            for d in np.linspace(1, 1e3, num=10) * 1e-9:
                for theta_outer in np.linspace(-np.pi/2 + 1e-3, np.pi/2 - 1e-3, num=20):
                    for n_outer in np.linspace(0.1, 10, num=10):
                        for n_layer in np.linspace(0.1, 10, num=10):
                            # for n_substrate in np.linspace(0.1, 10, num=10):
                            total_internal_reflection: bool = 1. < n_outer / n_layer * np.sin(np.abs(theta_outer))
                            n_substrate = n_outer

                            b_0_correct_s, b_0_correct_p = calculate_amplitudes_using_analytic_formulas(wavelength, d, n_outer, n_layer, n_substrate, theta_outer)

                            args = M, np.array([n_layer]), np.array([d]), wavelength, n_outer, n_substrate, theta_outer
                            b_0_to_be_tested_s = amplitude(0, *args)
                            b_0_to_be_tested_p = amplitude(1, *args)

                            for polarisation, b_0_correct, b_0_to_be_tested in ('s', b_0_correct_s, b_0_to_be_tested_s), ('p', b_0_correct_p, b_0_to_be_tested_p):
                                err_msg = f'polarisation: {polarisation}, wavelength: {wavelength}, d: {d}, theta_outer: {theta_outer}, n_outer: {n_outer}, n_layer: {n_layer}, total_internal_reflection: {total_internal_reflection}, ' + \
                                    f'b_0_correct: {b_0_correct, np.abs(b_0_correct) ** 2}, b_0_to_be_tested: {b_0_to_be_tested, np.abs(b_0_to_be_tested) ** 2}'
                                self.assertAlmostEqual(b_0_correct.real, b_0_to_be_tested.real, delta=1e-11, msg=err_msg)
                                self.assertAlmostEqual(b_0_correct.imag, b_0_correct.imag, delta=1e-11, msg=err_msg)

        # fig: plt.Figure
        # ax: plt.Axes
        # fig, ax = plt.subplots()
        # ax.plot(wavelengths, R_correct)
        # ax.plot(wavelengths, R_to_be_tested)
        # ax.legend()
        # # ax.set_ylim(0,1.1)
        # fig.show()

        # for correct, to_test in zip(R_correct, R_to_be_tested):
        #     self.assertAlmostEqual(correct, to_test, places=15)
        # theta = np.linspace(0,0.06,num=1000)
        # R1 = np.array([reflection_coefficient(np.sqrt(0.5), 400* 1e-9, n_layer, d[0], np.cos(t)) for t in theta])
        # R2 = np.array([reflection_coefficient(np.sqrt(0.7), 400* 1e-9, n_layer, d[0], np.cos(t)) for t in theta])
        # R3 = np.array([reflection_coefficient(np.sqrt(0.9), 400* 1e-9, n_layer, d[0], np.cos(t)) for t in theta])

    def test_antireflection_coating(self):
        # Digitised reflectivity against wavelength plot from paper
        df = pd.read_csv(Path('data') / 'digitised_plots_from_Hobson_Baldwin_2004_paper/plot_1_data.csv')

        wavelengths: np.ndarray = df.iloc[:, 0].values * 1e-9
        R_paper: np.ndarray = df.iloc[:, 1].values

        # Calculation from multilayer data
        df = pd.read_csv(Path('data') / 'Hobson_Baldwin_2004_anti_reflection_coating_data.txt')

        # Sort
        p = wavelengths.argsort()
        wavelengths = wavelengths[p]
        R_paper = R_paper[p]

        # Calculation from multilayer data
        M: int = df.shape[0]
        n: np.ndarray = df.iloc[:, 1].values
        d: np.ndarray = df.iloc[:, 2].values * 1e-9
        n_outer = 1.00
        n_substrate = 1.50
        theta_outer = 0.

        # wavelengths = np.linspace(600, 2300, 1000) * 1e-9
        R_s = np.array([reflectivity(0, M, n, d, lam, n_outer, n_substrate, theta_outer) for lam in wavelengths])
        R_p = np.array([reflectivity(1, M, n, d, lam, n_outer, n_substrate, theta_outer) for lam in wavelengths])

        # diff = R_paper - R
        # print(np.argmax(diff))

        # Compare
        np.testing.assert_allclose(R_paper, R_s, rtol=0, atol=0.0016)
        np.testing.assert_allclose(R_paper, R_p, rtol=0, atol=0.0016)

        # fig, ax = plt.subplots()
        # ax: plt.Axes
        # ax.plot(wavelengths, R_paper, label='paper')
        # ax.plot(wavelengths, R_s, label='s-polarisation')
        # ax.plot(wavelengths, R_p, '.', label='p-polarisation')
        # ax.set_xlabel(r'$\lambda_\mathrm{outer}$ [nm]')
        # ax.set_ylabel('R')
        # ax.legend()
        # plt.show()

    def test_against_plot_2_from_paper_for_dichroic(self):
        # Digitised reflectivity against wavelength plot from paper
        df = pd.read_csv(Path('data') / 'digitised_plots_from_Hobson_Baldwin_2004_paper/plot_2_data.csv')

        wavelengths: np.ndarray = df.iloc[:, 0].values * 1e-9
        R_paper: np.ndarray = df.iloc[:, 1].values

        # Sort
        p = wavelengths.argsort()
        wavelengths = wavelengths[p]
        R_paper = R_paper[p]

        # Calculation from multilayer data
        df = pd.read_csv(Path('data') / 'Hobson_Baldwin_2004_dichroic_data.txt')

        M: int = df.shape[0]
        n: np.ndarray = df.iloc[:, 1].values
        d: np.ndarray = df.iloc[:, 2].values * 1e-9
        n_outer = 1.00
        n_substrate = 1.50
        theta_outer = 0.

        # wavelengths = np.linspace(600, 2300, 1000) * 1e-9
        R_s = np.array([reflectivity(0, M, n, d, lam, n_outer, n_substrate, theta_outer) for lam in wavelengths])
        R_p = np.array([reflectivity(1, M, n, d, lam, n_outer, n_substrate, theta_outer) for lam in wavelengths])

        # Compare
        np.testing.assert_allclose(R_paper, R_s, rtol=0, atol=0.1443)
        np.testing.assert_allclose(R_paper, R_p, rtol=0, atol=0.1443)

        # fig, ax = plt.subplots()
        # ax: plt.Axes
        # ax.plot(wavelengths, R_paper, label='paper')
        # ax.plot(wavelengths, R_s, label='s-polarisation')
        # ax.plot(wavelengths, R_p, '.', label='p-polarisation')
        # ax.set_xlabel(r'$\lambda_\mathrm{outer}$ [nm]')
        # ax.set_ylabel('R')
        # ax.legend()
        # # ax.set_ylim(0, 1.2)
        # plt.show()