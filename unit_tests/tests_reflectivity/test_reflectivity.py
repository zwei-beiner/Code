from unittest import TestCase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import reflectivity_s


class Test_reflectivity(TestCase):
    def test_reflectivity_s(self):
        df = pd.read_csv('./data/Hobson_Baldwin_2004_anti_reflection_coating_data.txt')

        M: int = df.shape[0]
        n: np.ndarray = df.iloc[:,1].values
        d: np.ndarray = df.iloc[:,2].values
        n_outer = 1.00
        n_substrate = 1.50
        theta_outer = 0.

        wavelengths = np.linspace(600, 2300, 1000)
        R = np.array([reflectivity_s(
            M,
            n,
            d,
            lam * 1e-9,
            n_outer,
            n_substrate,
            theta_outer
        )
        for lam in wavelengths])

        print(R[0])

        fig, ax = plt.subplots()
        ax: plt.axes.Axes
        ax.plot(wavelengths, R)
        ax.set_ylim(0, 1.2)
        plt.show()


