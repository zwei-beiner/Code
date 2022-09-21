from typing import Union, Callable
from types import FunctionType

import numpy as np

from ConstraintTypes import AbstractConstraint, FixedConstraint, CategoricalConstraint, BoundedConstraint

# Define the refractive index type, which is a function from an array of wavelengths to refractive indices at these
# wavelengths.
RefractiveIndex = Callable[[np.ndarray], np.ndarray]

class n_constraints:
    """
    Class which handles a refractive index specification.
    - Parses the user input of the refractive index specification.
    - Checks the types of the inputs.
    - Enforces that a refractive index constraint can either be fixed or categorical.
    - Provides useful functions for the refractive indices input by the user.
    """

    def __init__(self, params: tuple[tuple[str, Union[RefractiveIndex, list[RefractiveIndex]]], ...]):
        """
        @param params: Refractive index specification
        """

        self._specification = params
        # Store an AbstractConstraint for each layer.
        self._n: list[AbstractConstraint] = []
        # Store which layers have fixed refractive indices.
        self._fixed_indices: list[int] = []
        # Store which layers have unfixed refractive indices.
        self._unfixed_indices: list[int] = []

        # Parse the layer specification and allow only FixedConstraint or CategoricalConstraint.
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
        """
        Returns True if the refractive indices of all layers are fixed.
        """
        return all(type(v) is FixedConstraint for v in self._n)

    def all_not_fixed(self) -> bool:
        """
        Returns True if all refractive indices are unfixed.
        """
        return all(type(v) is CategoricalConstraint for v in self._n)

    def to_numpy_array(self) -> np.ndarray:
        """
        If all refractive indices are fixed, return a numpy array containing all the values of the refractive indices.
        """
        if not self.all_fixed():
            raise Exception('Constraints are not fixed.')
        return np.array([c.value for c in self._n])

    def get_fixed_values(self):
        """
        Return the values of the refractive indices of the layers which are fixed.
        """
        return list(map(lambda f: f.value, filter(lambda c: type(c) is FixedConstraint, self._n)))

    def get_unfixed_values(self):
        """
        Return the values of the refractive indices of the layers which are unfixed.
        """
        return list(map(lambda f: f.value, filter(lambda c: type(c) is not FixedConstraint, self._n)))

    def get_values_from_indices(self, indices: list[int]) -> list[RefractiveIndex]:
        """
        Return the values of the refractive indices from a list of integers.
        This is used to map the output of the optimisation to the corresponding refractive indices.
        For example, we have the multilayer specificaton
        (('fixed', 1.), ('categorical', [2., 3.]), ('fixed', 4.), ('categorical', [5., 6., 7.]))
        With the input
        [0, 1, 0, 2]
        this function would return
        [1., 3., 4., 7.]

        @param indices: List of integers.
        @return: List of refractive indices.
        """

        res: list[RefractiveIndex] = []
        for n, i in zip(self._n, indices):
            if type(n) is FixedConstraint:
                res.append(n.value)
            elif type(n) is CategoricalConstraint:
                res.append(n.value[i])
            else:
                raise TypeError(f'Invalid type: {type(n)}')
        return res

    def get_fixed_indices(self) -> list[int]:
        """
        Return which layers have fixed refractive indices.
        """
        return self._fixed_indices

    def get_unfixed_indices(self) -> list[int]:
        """
        Return which layers have unfixed refractive indices.
        """
        return self._unfixed_indices

    def get_specification(self) -> tuple[tuple[str, Union[RefractiveIndex, list[RefractiveIndex]]], ...]:
        return self._specification


class d_constraints:
    """
    Class which handles a thickness specification.
    - Parses the user input of the thickness specification.
    - Checks the types of the inputs.
    - Enforces that a thickness constraint can either be fixed or bounded.
    - Provides useful functions for the thickness input by the user.
    """

    def __init__(self, params: tuple[tuple[str, Union[float, tuple[float]]], ...]):
        """
        @param params: Layer thicknesses specification
        """

        self._specification = params
        # Store an AbstractConstraint for each layer.
        self._d: list[AbstractConstraint] = []
        # Store which layers have fixed refractive thicknesses.
        self._fixed_indices: list[int] = []
        # Store which layers have unfixed thicknesses.
        self._unfixed_indices: list[int] = []

        # Parse the layer specification and allow only FixedConstraint or BoundedConstraint.
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
        """
        Returns True if the thicknesses of all layers are fixed.
        """
        return all(type(v) is FixedConstraint for v in self._d)

    def all_not_fixed(self) -> bool:
        """
        Returns True if all thicknesses are unfixed.
        """
        return all(type(v) is BoundedConstraint for v in self._d)

    def to_numpy_array(self) -> np.ndarray:
        """
        If all thicknesses are fixed, return a numpy array containing all the values of the thicknesses.
        """
        if not self.all_fixed():
            raise Exception('Constraints are not fixed.')
        return np.array([c.value for c in self._d])

    def get_fixed_values(self):
        """
        Return the values of the thicknesses of the layers which are fixed.
        """
        return list(map(lambda f: f.value, filter(lambda c: type(c) is FixedConstraint, self._d)))

    def get_unfixed_values(self):
        """
        Return the values of the thicknesses of the layers which are unfixed.
        """
        return list(map(lambda f: f.value, filter(lambda c: type(c) is not FixedConstraint, self._d)))

    def get_fixed_indices(self) -> list[int]:
        """
        Return which layers have fixed thicknesses.
        """
        return self._fixed_indices

    def get_unfixed_indices(self) -> list[int]:
        """
        Return which layers have unfixed thicknesses.
        """
        return self._unfixed_indices

    def get_specification(self):
        return self._specification

    def get_D_max(self) -> float:
        """
        Returns maximum possible thickness D_max.
        """

        # Sum all fixed values.
        sum_1 = sum(self.get_fixed_values())
        # Sum all upper limits taken from the tuples.
        sum_2 = sum(l[1] for l in self.get_unfixed_values())
        return sum_1 + sum_2


class Wavelength_constraint:
    """
    - Parses the user input on the wavelengths.
    - Enforces that the wavelengths can be either specified with a tuple (BoundedConstraint) or a numpy array (FixedConstraint)
    - Provides utility functions.
    """

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

    def get_values(self) -> Union[np.ndarray, tuple[float, float]]:
        return self._wavelengths.value

    def get_min_max(self) -> tuple[float, float]:
        if self.is_fixed():
            array = self.get_values()
            return np.amin(array), np.amax(array)
        else:
            return self.get_values()
