from pathlib import Path
from unittest import TestCase

import numpy as np
import numpy.random
import pandas
import collections
import matplotlib.pyplot as plt

import pypolychord
import pypolychord.settings
import pypolychord.priors
import scipy.optimize
import scipy.stats

from src.optimiser import output_results

class Test_reflectivity(TestCase):
    def test_2d_optimisation(self):
        # Expected optimum: (0, 0)

        def loglikelihood(theta):
            return np.log(1/np.sqrt(2 * np.pi)) - np.sum(theta ** 2)/2, []

        def prior(unit_cube):
            # Uniform prior
            return pypolychord.priors.UniformPrior(-5, 5)(unit_cube)

        nDims = 2

        output_results(loglikelihood, nDims, prior, 1)


    def test_4d_rosenbrock(self):
        # Expected global optimum: (1,1,1,1)
        # Local optimum: (-1,1,1,1)

        nDims = 2

        def loglikelihood(theta):
            rosenbrock = np.sum(100 * (theta[1:] - theta[:nDims-1] ** 2) ** 2 + (1 - theta[:nDims - 1]) ** 2)
            return -rosenbrock, []

        def loglikelihood_2d(theta):
            x = theta[0]
            y = theta[1]
            return -((1 - x) ** 2 + 100 * (y - x ** 2) ** 2), []

        # import numpy as np
        # import matplotlib.pyplot as plt
        # from matplotlib import ticker
        # a = 3
        # xlist = np.linspace(-a, a, 1000)
        # ylist = np.linspace(-a, a, 1000)
        # X, Y = np.meshgrid(xlist, ylist)
        # Z = -loglikelihood_2d(X, Y)
        # fig, ax = plt.subplots(1, 1)
        # # cp = ax.contourf(X, Y, Z, locator=ticker.LogLocator())
        # ax = plt.axes(projection='3d')
        # ax.contour3D(X, Y, Z, 50, cmap='autumn_r')
        # # fig.colorbar(cp)  # Add a colorbar to a plot
        # ax.set_title('Filled Contours Plot')
        # # ax.set_xlabel('x (cm)')
        # ax.set_ylabel('y (cm)')
        # ax.set_zlim(0, 200)
        # plt.show()

        # for x in np.linspace(-5,5,1000):
        #     for y in np.linspace(-5,5,1000):
        #         theta = np.array([x,y])
        #         self.assertAlmostEqual(loglikelihood(theta)[0], loglikelihood_2d(theta)[0], delta=1e-15)

        def prior(cube):
            return pypolychord.priors.UniformPrior(-5,5)(cube)

        output_results(loglikelihood, nDims, prior, 10)


    def test_uniorm_discrete_prior(self):
        """
        Test how to build a categorical sampler as a map from the (continuous) unit cube.
        """

        categories = ['a', 'b', 'c', 'd', 'e']

        def prior(x):
            rv = scipy.stats.randint(low=0, high=len(categories))
            index = rv.ppf(x)
            return categories[int(index)]

        # fig, ax = plt.subplots()
        # x = np.linspace(-1,2,1000)
        # rv = scipy.stats.randint(low=0, high=7)
        # y = prior(x)
        # ax.plot(x, y)
        # plt.show()

        rng = np.random.default_rng(0)
        samples = []
        for i in range(int(1e4)):
            rv = rng.uniform()
            samples.append(prior(rv))

        letter_counts = collections.Counter(samples)
        df = pandas.DataFrame.from_dict(letter_counts, orient='index')
        df.plot(kind='bar')
        plt.show()
