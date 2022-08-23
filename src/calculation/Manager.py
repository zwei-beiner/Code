from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Callable
from types import FunctionType

import numpy as np
import pypolychord
import scipy.stats

from pypolychord.priors import UniformPrior

import sys

from src.calculation.Utils import Utils

# Add subdirectory manually to sys.path. This is necessary because we can't place an __init__.py file into
# the subdirectory, as this breaks Cython (this is a known Cython bug)
# This enables us to import modules from the subdirectory directly, e.g. 'import reflectivity_c_file'
file_path = str(Path(__file__).parents[1] / 'calculation')
sys.path.insert(1, file_path)
from reflectivity_c_file import reflectivity, calculate_wavelengths, amplitude


class AbstractConstraint(ABC):
    def __init__(self, type):
        self._type = type

    @property
    @abstractmethod
    def value(self):
        pass

    @value.setter
    @abstractmethod
    def value(self, value):
        pass


class FixedConstraint(AbstractConstraint):
    def __init__(self, type, value):
        super().__init__(type)
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not (type(value) is self._type):
            raise TypeError(f'Incorrect type: {value}')
        self._value = value


class BoundedConstraint(AbstractConstraint):
    def __init__(self, type, value):
        super().__init__(type)
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not (type(value) is tuple and list(map(type, value)) == [self._type, self._type]):
            raise TypeError(f'Incorrect type: {value}')
        self._value = value


class CategoricalConstraint(AbstractConstraint):
    def __init__(self, type, value):
        super().__init__(type)
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not (type(value) is list and all(type(v) is self._type for v in value)):
            TypeError(f'Incorrect type: {value}')
        self._value = value


# class M_constraints:
#     def __init__(self, constraint: str, params):
#         if constraint == 'fixed':
#             self.constraint = FixedConstraint(int, params)
#         elif constraint == 'bounded':
#             self.constraint = BoundedConstraint(int, params)
#         else:
#             raise ValueError(f'Invalid input: {constraint, params}')

RefractiveIndex = Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]

class n_constraints:
    def __init__(self, params: tuple[tuple[str, Union[RefractiveIndex, list[RefractiveIndex]]], ...]):
        self._n: list[AbstractConstraint] = []
        self._fixed_indices: list[int] = []
        self._unfixed_indices: list[int] = []

        for i, (constraint, value) in enumerate(params):
            if constraint == 'fixed':
                self._n.append(FixedConstraint(FunctionType, value))
                self._fixed_indices.append(i)
            elif constraint == 'categorical':
                self._n.append(CategoricalConstraint(FunctionType, value))
                self._unfixed_indices.append(i)
            else:
                raise ValueError(f'Invalid input: {constraint, value}')
        # print([self._n[i].value for i in range(len(self._n))])

    def all_fixed(self) -> bool:
        return all(type(v) is FixedConstraint for v in self._n)

    def all_not_fixed(self) -> bool:
        return all(type(v) is CategoricalConstraint for v in self._n)

    def to_numpy_array(self) -> np.ndarray:
        if not self.all_fixed():
            raise Exception('Constraints are not fixed.')
        return np.array([c.value for c in self._n])

    def get_fixed_values(self):
        return list(map(lambda f: f.value, filter(lambda c: type(c) is FixedConstraint, self._n)))

    def get_unfixed_values(self):
        return list(map(lambda f: f.value, filter(lambda c: type(c) is not FixedConstraint, self._n)))

    def get_fixed_indices(self) -> list[int]:
        return self._fixed_indices

    def get_unfixed_indices(self) -> list[int]:
        return self._unfixed_indices


class d_constraints:
    def __init__(self, params: tuple[tuple[str, Union[float, tuple[float]]], ...]):
        self._d: list[AbstractConstraint] = []
        self._fixed_indices: list[int] = []
        self._unfixed_indices: list[int] = []

        for i, (constraint, value) in enumerate(params):
            if constraint == 'fixed':
                self._d.append(FixedConstraint(float, value))
                self._fixed_indices.append(i)
            elif constraint == 'bounded':
                self._d.append(BoundedConstraint(float, value))
                self._unfixed_indices.append(i)
            else:
                raise ValueError(f'Invalid input: {constraint}')

    def all_fixed(self) -> bool:
        return all(type(v) is FixedConstraint for v in self._d)

    def all_not_fixed(self) -> bool:
        return all(type(v) is BoundedConstraint for v in self._d)

    def to_numpy_array(self) -> np.ndarray:
        if not self.all_fixed():
            raise Exception('Constraints are not fixed.')
        return np.array([c.value for c in self._d])

    def get_fixed_values(self):
        return list(map(lambda f: f.value, filter(lambda c: type(c) is FixedConstraint, self._d)))

    def get_unfixed_values(self):
        return list(map(lambda f: f.value, filter(lambda c: type(c) is not FixedConstraint, self._d)))

    def get_fixed_indices(self) -> list[int]:
        return self._fixed_indices

    def get_unfixed_indices(self) -> list[int]:
        return self._unfixed_indices


class Wavelength_constraint:
    def __init__(self, wavelengths: Union[np.ndarray, tuple[float]]):
        if type(wavelengths) is np.ndarray:
            self._wavelengths = FixedConstraint(np.ndarray, wavelengths)
        elif type(wavelengths) is tuple:
            self._wavelengths = BoundedConstraint(float, wavelengths)

    def is_fixed(self) -> bool:
        if type(self._wavelengths) is FixedConstraint:
            return True
        elif type(self._wavelengths) is BoundedConstraint:
            return False
        else:
            raise ValueError(f'Invalid type: {type(self._wavelengths)}')

    def get_values(self) -> Union[np.ndarray, tuple[float]]:
        return self._wavelengths.value


# class n_constraints:
#     constraint_type = ConstraintType2
#
#     def __init__(self):


class Optimiser:
    def __init__(self, project_name: str,
                 M: int,
                 n_outer: RefractiveIndex,
                 n_substrate: RefractiveIndex,
                 theta_outer: float,
                 wavelengths: Union[np.ndarray, tuple[float, float]],
                 n_specification: tuple[tuple[str, Union[RefractiveIndex, list[RefractiveIndex]]], ...],
                 d_specification: tuple[tuple[str, Union[float, tuple[float, float]]], ...],
                 p_pol_weighting: float,
                 s_pol_weighting: float,
                 phase_weighting: float,
                 target_reflectivity_s: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros(len(x)),
                 target_reflectivity_p: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros(len(x)),
                 target_relative_phase: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros(len(x)),
                 weight_function_s: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones(len(x)),
                 weight_function_p: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones(len(x)),
                 weight_function_phase: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones(len(x))):

        self._project_name = project_name

        # Default values
        # self.set_constraint_M('bounded', (1, 200))
        self._M = M
        self._n_outer = n_outer
        self._n_substrate = n_substrate
        self._theta_outer = theta_outer
        self._wavelengths = Wavelength_constraint(wavelengths)

        if len(n_specification) is not self._M or len(d_specification) is not self._M:
            raise Exception(f'Invalid specification length: {len(n_specification), len(d_specification)}')
        self._n_constraints = n_constraints(n_specification)
        self._d_constraints = d_constraints(d_specification)

        self._s_pol_weighting = s_pol_weighting
        self._p_pol_weighting = p_pol_weighting
        self._phase_weighting = phase_weighting

        self._target_reflectivity_s = target_reflectivity_s
        self._target_reflectivity_p = target_reflectivity_p
        self._target_relative_phase = target_relative_phase

        self._weight_function_s = weight_function_s
        self._weight_function_p = weight_function_p
        self._weight_function_phase = weight_function_phase


    # def set_constraint_M(self, M: int):
    #     if not(type(M) is int):
    #         raise TypeError(f'Incorrect type: {M}')
    #     self._M = M
    #     # self._M_constraints = M_constraints(**locals())

    # def set_constraint_n(self, params):
    #
    #
    # def set_constraint_d(self, params):
    #

    # def set_wavelengths(self, wavelengths: np.ndarray):
    #     self._wavelengths = Wavelength_constraint(wavelengths)

    def _build_amplitude_function(self):
        split = len(self._n_constraints.get_unfixed_indices())
        nDims = split + len(self._d_constraints.get_unfixed_indices())

        # params is an array of length 2M-nFixed where nFixed is the total number of fixed parameters.
        def amplitude_wrapper(params: np.ndarray, wavelength: float, polarisation: int):
            n = np.zeros(self._M)
            n[self._n_constraints.get_fixed_indices()] = [n(wavelength) for n in self._n_constraints.get_fixed_values()]
            n[self._n_constraints.get_unfixed_indices()] = [ns[np.int_(params[:split][i])](wavelength) for i, ns in enumerate(self._n_constraints.get_unfixed_values())]

            d = np.zeros(self._M)
            d[self._d_constraints.get_fixed_indices()] = self._d_constraints.get_fixed_values()
            d[self._d_constraints.get_unfixed_indices()] = params[split:]

            return amplitude(polarisation, self._M, n, d, wavelength, self._n_outer(wavelength), self._n_substrate(wavelength),
                             self._theta_outer)

        n_unfixed_values: list[list[RefractiveIndex]] = self._n_constraints.get_unfixed_values()
        d_unfixed_values: list[tuple[float]] = self._d_constraints.get_unfixed_values()
        samplers = [Utils.categorical_sampler(n_unfixed_values[i]) for i in range(0, split)] + [
            Utils.uniform_sampler(*d_unfixed_values[i]) for i in range(0, nDims - split)]

        def prior(unit_cube):
            sample = np.zeros(nDims)

            for i in range(0, split):
                sample[i] = samplers[i](unit_cube[i])

            for i in range(split, nDims):
                sample[i] = samplers[i](unit_cube[i])

            return sample

        return amplitude_wrapper, prior


    def _build_merit_function_and_prior(self):
        amplitude_wrapper, prior = self._build_amplitude_function()

        def merit_function_wrapper(params: np.ndarray, wavelengths: np.ndarray):
            num_wavelengths = len(wavelengths)

            reflectivities_s = np.zeros(num_wavelengths)
            reflectivities_p = np.zeros(num_wavelengths)
            relative_phases = np.zeros(num_wavelengths)
            for i in range(num_wavelengths):
                amplitude_s = amplitude_wrapper(params, wavelengths[i], 0)
                amplitude_p = amplitude_wrapper(params, wavelengths[i], 1)

                reflectivities_s[i] = np.abs(amplitude_s) ** 2
                reflectivities_p[i] = np.abs(amplitude_p) ** 2
                relative_phases[i] = np.angle(amplitude_s) - np.angle(amplitude_p)

            target_reflectivities_s = self._target_reflectivity_s(wavelengths)
            target_reflectivities_p = self._target_reflectivity_p(wavelengths)
            target_relative_phase = self._target_relative_phase(wavelengths)

            weights_s = self._weight_function_s(wavelengths)
            weights_p = self._weight_function_p(wavelengths)
            weights_relative_phase = self._weight_function_phase(wavelengths)

            return self._s_pol_weighting * np.mean(((reflectivities_s - target_reflectivities_s) / weights_s) ** 2) + \
                   self._p_pol_weighting * np.mean(((reflectivities_p - target_reflectivities_p) / weights_p) ** 2) + \
                   self._phase_weighting * np.mean(((relative_phases - target_relative_phase) / weights_relative_phase) ** 2)

        if self._wavelengths.is_fixed():
            def merit_function(params: np.ndarray):
                wavelengths = self._wavelengths.get_values()
                return merit_function_wrapper(params, wavelengths)

        else:
            def merit_function(params: np.ndarray):
                split = len(self._n_constraints.get_unfixed_indices())
                wavelengths = calculate_wavelengths(*self._wavelengths.get_values(),
                                np.array(self._d_constraints.get_fixed_values()).sum() + params[split:].sum())
                return merit_function_wrapper(params, wavelengths)

        return merit_function, prior


    def _build_likelihood_and_prior(self):
        merit_function, prior = self._build_merit_function_and_prior()

        def likelihood(params):
            return -merit_function(params), []

        return likelihood, prior


    def run(self):
        likelihood, prior = self._build_likelihood_and_prior()
        nDims = len(self._n_constraints.get_unfixed_indices()) + len(self._d_constraints.get_unfixed_indices())
        nDerived = 0
        niter = 10

        settings = pypolychord.settings.PolyChordSettings(nDims, nDerived)
        settings.nlive = 10 * settings.nlive
        settings.read_resume = False  # TODO: Check if directory exists
        settings.maximise = True
        settings.max_ndead = int(niter * nDims * settings.nlive)  # TODO: Check if likelihood converged
        settings.precision_criterion = -1
        settings.feedback = 3
        settings.base_dir = str(Path(__file__).parent / 'polychord_output')

        output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
        print('PolyChord run completed.')

        with open(Path(settings.base_dir) / (settings.file_root + '.maximum'), 'r') as file:
            lines = file.readlines()

        params = lines[3].split()

        with open(str(Path(__file__).parent / 'optimal_parameters.txt'), 'w') as file:
            file.write('\n'.join(map(str, params)))


# if __name__ == '__main__':
    # Optimiser(4, compile=False)

    # c = FixedConstraint(str, 1.9)
    # print(c.value)

    # # args = [None] * 8
    # args = {'polarisation': None, 'M': None, 'n': [None] * self._M, 'd': [None] * self._M,
    #         'n_outer': None, 'n_substrate': None, 'theta_outer': None}
    #
    # args['polarisation'] = self._polarisation
    # args['M'] = self._M
    # # if self._wavelengths is None:
    # #     args['wavelengths'] = calculate_wavelengths(600 * 1e-9, 2300 * 1e-9)
    # # else:
    # #     args['wavelengths'] = self._wavelengths
    # args['n_outer'] = self._n_outer
    # args['n_substrate'] = self._n_substrate
    # args['theta_outer'] = self._theta_outer
    #
    #

    # def reflectivity1(n, d, wavelength):
    #     return reflectivity(0, M, n, d, wavelength, n_outer, n_substrate, theta_outer)

    # if self._n_constraints.all_fixed() and self._d_constraints.all_fixed():
    #     raise Exception('All variables are fixed.')
    #
    # elif self._n_constraints.all_fixed() and self._d_constraints.all_not_fixed():
    #     args['n'] = self._n_constraints.to_numpy_array()
    #     # n = self._n_constraints.to_numpy_array()
    #     return lambda d, wavelength, M=args['M'], n=args['n'], n_outer=args['n_outer'], n_substrate=args['n_substrate']: \
    #         reflectivity(M, n, d, wavelength, n_outer, n_substrate)

    # elif self._d_constraints.all_fixed() and self._n_constraints.all_not_fixed():
    #     args['d'] = self._d_constraints.to_numpy_array()
    #     # d = self._d_constraints.to_numpy_array()
    #     # def reflectivity2(n, wavelength):
    #     #     return reflectivity1(n, d, wavelength)
    #     # return reflectivity2
    #
    # else:
