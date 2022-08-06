import time
from pathlib import Path
from unittest import TestCase

import numpy as np
import numpy.typing as npt
import numpy.random
import pandas
import collections
import matplotlib.pyplot as plt

import pypolychord
import pypolychord.settings
import pypolychord.priors
import scipy.optimize
import scipy.stats

from src.reflectivity import reflectivity
from src.optimiser import output_results


class Test_reflectivity(TestCase):
    def test_if_Hobson_Baldwin_antireflection_coating_results_are_reproduced(self):
        M = 25
        def make_merit_function(n: npt.NDArray[np.float_], wavelengths: npt.NDArray[np.float_], target_reflecitivities: npt.NDArray[np.float_]):
            theta_outer = np.float_(0)
            polarisation = 0
            n_outer = np.float_(1.00)
            n_substrate = np.float_(1.50)

            delta_R = 0.01

            def chi_squared(d):
                reflectivities = np.array(
                    [reflectivity(polarisation, M, n, d, wavelength, n_outer, n_substrate, theta_outer)
                     for wavelength in wavelengths]
                )
                return np.sum(((reflectivities - target_reflecitivities) / delta_R) ** 2)
            return chi_squared

        # Make wavelengths
        wavelengths = np.concatenate((np.arange(600, 1000, 20), np.arange(1000, 2000, 40), np.arange(2000, 2300, 50), np.array([2300]))) * 1e-9
        target_reflectivities = np.zeros(len(wavelengths))
        assert len(wavelengths) == 52

        # Make n
        n = np.zeros(M-1)
        n[::2] = 1.45
        n[1::2] = 2.10
        n = np.concatenate((np.array([1.37]), n))
        assert len(n) == 25

        chi_squared = make_merit_function(n, wavelengths, target_reflectivities)

        d_min = 0
        d_max = 350 * 1e-9
        def prior(unit_cube):
            return pypolychord.priors.UniformPrior(d_min, d_max)(unit_cube)

        def loglikelihood(theta):
            return -chi_squared(theta), []

        tic = time.process_time()
        output_results(loglikelihood, M, prior, 0)
        toc = time.process_time()
        print(f'Elapsed time = {toc - tic} seconds.')
