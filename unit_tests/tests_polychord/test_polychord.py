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

import anesthetic
import anesthetic.utils
from anesthetic.gui.plot import RunPlotter

# from src.optimiser import output_results, plot


class Test_reflectivity(TestCase):
    def test_2d_optimisation(self):
        # Expected optimum: (0, 0)

        def loglikelihood(theta):
            # Note: Loglikelihood can simply be the likelihood when we are only seeking the maximum.
            # Test maximisation of negative function.
            return np.exp(-np.sum(theta ** 2)/2) - 30, []

        def prior(unit_cube):
            # Uniform prior
            return pypolychord.priors.UniformPrior(-5, 5)(unit_cube)

        nDims = 3

        nDerived = 0
        niter = 20

        settings = pypolychord.settings.PolyChordSettings(nDims, nDerived)
        settings.nlive = 10 * settings.nlive
        settings.read_resume = False
        # settings.maximise = True
        settings.max_ndead = int(niter * nDims * settings.nlive)  # TODO: Check if likelihood converged
        print(f'Maximum number of dead points: {settings.max_ndead}')
        settings.precision_criterion = -1
        settings.feedback = 3
        settings.base_dir = str('polychord_output')

        pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings, prior)
        print('PolyChord run completed.')

        samples = anesthetic.NestedSamples(root=str(Path(settings.base_dir) / settings.file_root))
        np.random.seed(71)
        print(samples.d())
        samples._compute_insertion_indexes()
        ks = anesthetic.utils.insertion_p_value(samples.insertion, settings.nlive)
        print(ks['p-value'])


    def test_4d_rosenbrock(self):
        # Expected global optimum: (1,1,1,1)
        # Local optimum: (-1,1,1,1)

        nDims = 4

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

        settings = output_results(loglikelihood, nDims, prior, 13, True)
        plot(settings)




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
