from unittest import TestCase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import reflectivity_s


class Test_reflectivity(TestCase):

    @classmethod
    def _make_test_matrix_for_one_layer(cls, M: np.int_, n: np.ndarray, d: np.ndarray, k_outer: np.float_, n_outer: np.float_, n_substrate: np.float_,
                 theta_outer: np.float_) -> np.ndarray:
        return



    def test_thin_film_interference(self):
        M: int = 1
        d: np.ndarray = np.array([1]) * 1e-3
        n_outer = 1.00
        n_layer = 1.50
        n: np.ndarray = np.array([n_layer])
        n_substrate = n_outer
        theta_outer = 0.

        # Fresnel coefficients
        # def calc_k_x(n):
        #     return k_outer * np.sqrt((n / n_outer) ** 2 - np.sin(theta_outer) ** 2)
        # r_12: np.complex_ = (k_x_1 - k_x_2) / (k_x_1 + k_x_2)
        # t_12: np.complex_ = 2 * k_x_1 / (k_x_1 + k_x_2)

        # Angles in layer and substrate
        def calc_cos_theta_i(n_i):
            # Snell's law
            sin_theta_i = n_outer * np.sin(theta_outer) / n_i
            # Note: Cosine can be complex
            cos_theta_i = np.sqrt(1-sin_theta_i ** 2)
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

        self.assertAlmostEqual(r_12.real, -r_23.real, places=15)
        self.assertAlmostEqual(r_12.imag, -r_23.imag, places=15)

        def reflection_coefficient(r, lambda_0, n, d, cos_theta):
            phi = 2 * np.pi / lambda_0 * n * d * cos_theta
            return 4 * (r ** 2) * (np.sin(phi) ** 2) / ((1 - r ** 2) ** 2 + 4 * (r ** 2) * (np.sin(phi) ** 2))

        wavelengths = np.linspace(1, 10, num=10000) * 1e-6
        R_correct = np.array([reflection_coefficient(r_12, lam, n_layer, d[0], cos_theta_layer) for lam in wavelengths])
        R_to_be_tested = np.array([reflectivity_s(M, n, d, lam, n_outer, n_substrate, theta_outer) for lam in wavelengths])

        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots()
        ax.plot(wavelengths, R_correct)
        ax.plot(wavelengths, R_to_be_tested)
        ax.legend()
        # ax.set_ylim(0,1.1)
        fig.show()

        self.assertEqual(R_correct.shape[0], R_to_be_tested.shape[0])
        for correct, to_test in zip(R_correct, R_to_be_tested):
            self.assertAlmostEqual(correct, to_test, places=15)
        # theta = np.linspace(0,0.06,num=1000)
        # R1 = np.array([reflection_coefficient(np.sqrt(0.5), 400* 1e-9, n_layer, d[0], np.cos(t)) for t in theta])
        # R2 = np.array([reflection_coefficient(np.sqrt(0.7), 400* 1e-9, n_layer, d[0], np.cos(t)) for t in theta])
        # R3 = np.array([reflection_coefficient(np.sqrt(0.9), 400* 1e-9, n_layer, d[0], np.cos(t)) for t in theta])



    def test_antireflection_coating(self):
        df = pd.read_csv('./data/Hobson_Baldwin_2004_anti_reflection_coating_data.txt')

        M: int = df.shape[0]
        n: np.ndarray = df.iloc[:,1].values
        d: np.ndarray = df.iloc[:,2].values * 1e-9
        n_outer = 1.00
        n_substrate = 1.50
        theta_outer = 0.
        # print(n,d)

        wavelengths = np.linspace(600, 2300, 1000) * 1e-9
        R = np.array([reflectivity_s(
            M,
            n,
            d,
            lam,
            n_outer,
            n_substrate,
            theta_outer
        )
        for lam in wavelengths])

        fig, ax = plt.subplots()
        # figsize=(12, 9), dpi=600
        ax: plt.Axes
        ax.plot(wavelengths * 1e9, R)
        ax.set_xlabel(r'$\lambda_\mathrm{outer}$ [nm]')
        ax.set_ylabel('R')
        # ax.set_ylim(0, 1.2)
        plt.show()

    def test_dichroic(self):
        df = pd.read_csv('./data/Hobson_Baldwin_2004_dichroic_data.txt')

        M: int = df.shape[0]
        n: np.ndarray = df.iloc[:, 1].values
        print(df.iloc[:, 2].values, type())
        d: np.ndarray = df.iloc[:, 2].values * 1e-9
        n_outer = 1.00
        n_substrate = 1.50
        theta_outer = 0.

        wavelengths = np.linspace(600, 2300, 1000) * 1e-9
        R = np.array([reflectivity_s(
            M,
            n,
            d,
            lam,
            n_outer,
            n_substrate,
            theta_outer
        )
        for lam in wavelengths])

        fig, ax = plt.subplots()
        # figsize=(12, 9), dpi=600
        ax: plt.Axes
        ax.plot(wavelengths * 1e9, R)
        ax.set_xlabel(r'$\lambda_\mathrm{outer}$ [nm]')
        ax.set_ylabel('R')
        # ax.set_ylim(0, 1.2)
        plt.show()