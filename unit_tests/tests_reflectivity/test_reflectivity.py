from unittest import TestCase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import reflectivity_s


class Test_reflectivity(TestCase):
    def test_thin_film_interference(self):
        M: int = 1
        d: np.ndarray = np.array([50]) * 1e-9
        n_outer = 1.00
        n_layer = 2.0
        n: np.ndarray = np.array([n_layer])
        n_substrate = 1.00
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

        self.assertAlmostEqual(r_12.real, -r_23.real, places=7)
        self.assertAlmostEqual(r_12.imag, -r_23.imag, places=7)

        def reflection_coefficient(r, lambda_0, n, d, cos_theta):
            phi = 2 * np.pi / lambda_0 * n * d * cos_theta
            return 4 * (r ** 2) * (np.sin(phi) ** 2) / ((1 - r ** 2) ** 2 + 4 * (r ** 2) * (np.sin(phi) ** 2))






    def test_reflectivity_s(self):
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

        print(R[0])

        fig, ax = plt.subplots()
        ax: plt.Axes
        ax.plot(wavelengths * 1e9, R)
        ax.set_xlabel(r'$\lambda_\mathrm{outer}$ [nm]')
        ax.set_ylabel('R')
        # ax.set_ylim(0, 1.2)
        plt.show()


