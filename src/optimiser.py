from pathlib import Path

import numpy as np
import numpy.typing as npt

# def merit_function(wavelength_min, wavelength_max):
#     """
#     Function to be minimised.
#     """
#
# def _make_wavelengths(wavelength_min, wavelength_max, d):
#     total_thickness = np.sum(d)
import pypolychord
import scipy.optimize


def output_results(loglikelihood, nDims, prior, niter, resume_from_previous_run: bool = False):
    """
    Maximises the function loglikelihood.

    @param loglikelihood: Function to be maximised
    @param nDims:
    @param prior:
    @param niter:
    @return:
    """
    nDerived = 0

    settings = pypolychord.settings.PolyChordSettings(nDims, nDerived)
    settings.read_resume = resume_from_previous_run
    settings.maximise = True
    settings.max_ndead = int(niter * nDims * settings.nlive)
    settings.precision_criterion = -1
    settings.feedback = 3
    settings.base_dir = 'polychord_output'

    output = pypolychord.run_polychord(loglikelihood, nDims, nDerived, settings, prior)
    print('PolyChord run completed.')

    with open(Path(settings.base_dir) / (settings.file_root + '.maximum'), 'r') as file:
        lines = file.readlines()

    # Split 4th line on spaces and cast each string to a float
    params = np.array(list(map(float, lines[3].split())))
    # params = lines[3].split()

    res = scipy.optimize.minimize(fun=lambda x: -loglikelihood(x)[0], x0=params, method='Powell')
    params = res.x
    # txt = res.message
    # print(params)

    print('Local optimisation completed.')

    with open('optimal_parameters.txt', 'w') as file:
        file.write(','.join(map(str, params)))
        # file.write('\n' + txt)

# def prior(unit_cube):
#     """
#     Maps uniform samples from the unit hypercube of dimension 2M+1 (i.e. dim(n)+dim(d)+dim(M)) to samples from physical space.
#
#     @param unit_cube: Random vector from U(0,1)^(2M+1)
#     @return:
#     """
#
# def categorical_samples():
