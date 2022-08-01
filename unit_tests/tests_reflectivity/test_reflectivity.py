from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import reflectivity, _make_matrix


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

    def test_make_matrix(self):
        """
        Tests if matrix is constructed correctly for a single layer (which is a Fabry-Perot interferometer).
        """

        def make_test_matrix_for_fabry_pelot_etalon(n: np.float_, d: np.float_, k_outer: np.float_,
                                                    n_outer: np.float_, n_substrate: np.float_,
                                                    theta_outer: np.float_) -> np.ndarray:
            k_x_outer: np.complex_ = np.complex_(k_outer * np.cos(theta_outer))
            k_x_layer: np.complex_ = np.complex_(k_outer * np.emath.sqrt((n / n_outer) ** 2 - np.sin(theta_outer) ** 2))
            # return np.array([(n / n_outer) ** 2])
            k_x_substrate: np.complex_ = k_outer * np.emath.sqrt(np.complex_((n_substrate / n_outer) ** 2 - np.sin(theta_outer) ** 2))

            phi_0: np.complex_ = k_x_layer * np.complex_(d)

            # mat = np.array([
            #     [-1, 1, np.exp(1j * phi_0), 0],
            #     [1, k_x_layer / k_x_outer, -k_x_layer / k_x_outer * np.exp(1j * phi_0), 0],
            #     [0, -np.exp(1j * phi_0), -1, 1],
            #     [0, -np.exp(1j * phi_0) * k_x_layer, k_x_layer, k_x_substrate]
            # ])
            mat = np.array([
                [0,     0,                                  np.exp(1j * phi_0),                             0],
                [0,     1,                                  -k_x_layer / k_x_outer * np.exp(1j * phi_0),    1],
                [-1,    k_x_layer / k_x_outer,              -1,                                             k_x_substrate],
                [1,     -np.exp(1j * phi_0),                k_x_layer,                                      0],
                [0,     -np.exp(1j * phi_0) * k_x_layer,    0,                                              0]
            ], dtype=np.complex_)
            return mat

        for n in np.linspace(0.1, 10, num=10):
            print(n)
            for d in np.linspace(1, 1e3, num=10) * 1e-9:
                for wavelength in np.linspace(200, 3000, num=10) * 1e-9:
                    for n_outer in np.linspace(0.1, 10, num=10):
                        for n_substrate in np.linspace(0.1, 10, num=10):
                            for theta in np.linspace(0, np.pi / 2 - 1e-3, num=10):
                                R_correct: np.ndarray = make_test_matrix_for_fabry_pelot_etalon(
                                    n, d, 2 * np.pi / wavelength, n_outer, n_substrate, theta
                                )

                                M = 1
                                R_to_be_tested: np.ndarray = _make_matrix(
                                    0, M, np.array([n]), np.array([d]), 2 * np.pi / wavelength, n_outer, n_substrate, theta
                                )

                                err_msg: str = f'n: {n}, d: {d}, wavelength: {wavelength}, n_outer: {n_outer}, ' \
                                               f'n_substrate: {n_substrate}, theta: {theta}' + \
                                               '\nR_correct:'+str(R_correct.tolist())+\
                                               '\n R_to_be_tested:'+str(R_to_be_tested.tolist())
                                np.testing.assert_allclose(
                                    R_correct, R_to_be_tested, rtol=0, atol=3e-5,
                                    err_msg=err_msg)
                                # np.testing.assert_allclose(
                                #     R_correct, R_to_be_tested, rtol=0, atol=1e-15,
                                #     err_msg=err_msg)

    def test_fabry_pelot_etalon(self):
        # Fresnel coefficients
        # def calc_k_x(n):
        #     return k_outer * np.sqrt((n / n_outer) ** 2 - np.sin(theta_outer) ** 2)
        # r_12: np.complex_ = (k_x_1 - k_x_2) / (k_x_1 + k_x_2)
        # t_12: np.complex_ = 2 * k_x_1 / (k_x_1 + k_x_2)

        def calculate_reflectivity_using_analytic_formulas(wavelength: np.float_, d: np.float_, n_outer: np.float_, n_layer: np.float_, n_substrate: np.float_, theta_outer: np.float_):
            """

            @param wavelength: wavelength in the outer medium. Hence wavelength in vacuum is "wavelength/n_outer"
            @param d: thickness of the layer in meters
            @param n_outer: refractive index of the outer medium
            @param n_layer: refractive index of the inner medium
            @param n_substrate: refractive index of the outer medium. Must be the same as the n_outer
            @param theta_outer: angle of incidence, i.e. angle to the normal in the outer medium. Can be in the range -pi/2<theta_outer<pi/2
            """

            # Angles in layer and substrate
            def calc_cos_theta_i(n_i):
                # Snell's law
                sin_theta_i = n_outer * np.sin(theta_outer) / n_i
                # Note: Cosine can be complex
                cos_theta_i = np.emath.sqrt(1 - sin_theta_i ** 2)
                return cos_theta_i

            cos_theta_outer = np.cos(theta_outer)
            cos_theta_layer = calc_cos_theta_i(n_layer)
            cos_theta_substrate = calc_cos_theta_i(n_substrate)

            # Fresnel coefficents for s polarisation (TE polarisation)
            def r_ij(n_i, cos_theta_i, n_j, cos_theta_j):
                return (n_i * cos_theta_i - n_j * cos_theta_j) / (n_i * cos_theta_i + n_j * cos_theta_j)

            def t_ij(n_i, cos_theta_i, n_j, cos_theta_j):
                return 2 * n_i * cos_theta_i / (n_i * cos_theta_i + n_j * cos_theta_j)

            r_12: np.complex_ = r_ij(n_outer, cos_theta_outer, n_layer, cos_theta_layer)
            t_12: np.complex_ = t_ij(n_outer, cos_theta_outer, n_layer, cos_theta_layer)

            r_23: np.complex_ = r_ij(n_layer, cos_theta_layer, n_substrate, cos_theta_substrate)
            t_23: np.complex_ = t_ij(n_layer, cos_theta_layer, n_substrate, cos_theta_substrate)

            self.assertEqual(n_outer, n_substrate)
            self.assertAlmostEqual(r_12.real, -r_23.real, delta=1e-4, msg='Refractive index of the outer medium and the substrate must be the same.')
            self.assertAlmostEqual(r_12.imag, -r_23.imag, delta=1e-4, msg='Refractive index of the outer medium and the substrate must be the same.')

            # def reflection_coefficient(r, lambda_0, n, d, cos_theta):
            r = r_12
            phi = 2 * np.pi / wavelength * n_layer * d * cos_theta_layer / n_outer
            amplitude = r * (1-np.exp(-2 * 1j * phi)) / (1 - r ** 2 * np.exp(-2 * 1j * phi))
            return np.abs(amplitude) ** 2
            # return 4 * (r ** 2) * (np.sin(phi) ** 2) / ((1 - r ** 2) ** 2 + 4 * (r ** 2) * (phi ** 2))

        M: int = 1
        # d: np.float_ = np.float_(1 * 1e-3)
        # n_outer: np.float_ = np.float_(1.00)
        # n_layer: np.float_ = np.float_(1.50)
        # n: np.ndarray = np.array([n_layer])
        # n_substrate = n_outer
        # theta_outer: np.float_ = np.float_(0.)

        for wavelength in np.linspace(200, 3000, num=10) * 1e-9:
            for d in np.linspace(1, 1e3, num=10) * 1e-9:
                for theta_outer in np.linspace(-np.pi/2 + 1e-6, np.pi/2 - 1e-6, num=20):
                    for n_outer in np.linspace(0.1, 10, num=10):
                        for n_layer in np.linspace(0.1, 10, num=10):
                            # for n_substrate in np.linspace(0.1, 10, num=10):
                            total_internal_reflection: bool = 1. < n_outer / n_layer * np.sin(np.abs(theta_outer))
                            n_substrate = n_outer
                            R_correct = calculate_reflectivity_using_analytic_formulas(wavelength, d, n_outer, n_layer, n_substrate, theta_outer)
                            R_to_be_tested = reflectivity(0, M, np.array([n_layer]), np.array([d]), wavelength, n_outer, n_substrate, theta_outer)
                            err_msg = f'wavelength: {wavelength}, d: {d}, theta_outer: {theta_outer}, n_outer: {n_outer}, n_layer: {n_layer}, total_internal_reflection: {total_internal_reflection}, ' + \
                                f'R_correct: {R_correct}, R_to_be_tested: {R_to_be_tested}'
                            self.assertAlmostEqual(R_correct, R_to_be_tested, delta=1e-9, msg=err_msg)
            # np.testing.assert_allclose(R_correct, R_to_be_tested, rtol=0, atol=1e-15)

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
        R = np.array([reflectivity(
            0,
            M,
            n,
            d,
            lam,
            n_outer,
            n_substrate,
            theta_outer
        )
            for lam in wavelengths])

        # diff = R_paper - R
        # print(np.argmax(diff))

        # Compare
        np.testing.assert_allclose(R_paper, R, rtol=0, atol=0.0016)

        # fig, ax = plt.subplots()
        # ax: plt.Axes
        # ax.plot(wavelengths, R_paper)
        # ax.plot(wavelengths, R)
        # ax.set_xlabel(r'$\lambda_\mathrm{outer}$ [nm]')
        # ax.set_ylabel('R')
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
        R = np.array([reflectivity(
            0,
            M,
            n,
            d,
            lam,
            n_outer,
            n_substrate,
            theta_outer
        )
            for lam in wavelengths])

        # fig, ax = plt.subplots()
        # # figsize=(12, 9), dpi=600
        # ax: plt.Axes
        # # ax.plot(wavelengths, R_paper, label='R_paper')
        # ax.plot(wavelengths, R, label='R_calculated')
        # ax.set_xlabel(r'$\lambda_\mathrm{outer}$ [nm]')
        # ax.set_ylabel('R')
        # ax.legend()
        # # ax.set_ylim(0, 1.2)
        # plt.show()

        # Compare
        np.testing.assert_allclose(R_paper, R, rtol=0, atol=0.1443)

    # def test_against_plot_1_from_paper_for_antireflection_coating(self):
    #     df = pd.read_csv('./data/digitised_plots_from_Hobson_Baldwin_2004_paper/plot_1_data.csv')
    #
    #     wavelengths: np.ndarray = df.iloc[:,0].values * 1e-9
    #     R_paper: np.ndarray = df.iloc[:,1].values
    #
    #     p = wavelengths.argsort()
    #     wavelengths = wavelengths[p]
    #     R_paper = R_paper[p]
    #
    #
    #
    #     # fig, ax = plt.subplots()
    #     # ax: plt.Axes
    #     # ax.plot(wavelengths, R)
    #     # plt.show()

    # def test_against_plot_2_from_paper_for_dichroic(self):
    #     df = pd.read_csv('./data/digitised_plots_from_Hobson_Baldwin_2004_paper/plot_2_data.csv')
    #
    #     wavelengths: np.ndarray = df.iloc[:,0].values
    #     R: np.ndarray = df.iloc[:,1].values
    #
    #     p = wavelengths.argsort()
    #     wavelengths = wavelengths[p]
    #     R = R[p]
    #
    #     fig, ax = plt.subplots()
    #     ax: plt.Axes
    #     ax.plot(wavelengths, R)
    #     plt.show()
