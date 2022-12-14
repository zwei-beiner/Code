import numpy as np
from WrapperClasses import RefractiveIndex


class Utils:
    @staticmethod
    def multilayer_specification(M: int, constraint: str, pattern: list):
        # "Tiles" a tuple with pattern.
        # E.g., returns ((constraint, pattern[0]), (constraint, pattern[1]), (constraint, pattern[0]),...) if len(pattern)==2
        res = tuple((constraint, pattern[i % len(pattern)]) for i in range(M))
        return res


    @staticmethod
    def take_layers_out(layer_specification: tuple[tuple], indices: list[int]) -> tuple[tuple]:
        """
        Removes layers in 'indices' from the 'layer_specification'.
        """
        return tuple(t for i, t in enumerate(layer_specification) if i not in indices)


    @staticmethod
    def constant(val: float) -> RefractiveIndex:
        """
        Returns a constant function.
        """

        def n(wavelengths: np.ndarray) -> np.ndarray:
            return np.full(len(wavelengths), val)
        return n


    @staticmethod
    def sellmeier_equation(Bs: list[float], Cs: list[float]) -> RefractiveIndex:
        """
        Implements the Sellmeier equation with arbitrary number of B's and C's.
        """

        Bs = np.array(Bs)
        Cs = np.array(Cs)
        if len(Bs) != len(Cs):
            raise ValueError

        def _n(wavelength: float) -> float:
            wavelength_squared = wavelength ** 2
            return np.sqrt(1 + np.sum(Bs * wavelength_squared / (wavelength_squared - Cs)))

        def n(wavelengths: np.ndarray) -> np.ndarray:
            return np.array([_n(w) for w in wavelengths])

        return n


    # @staticmethod
    # def categorical_sampler(categories: list):
    #     """
    #     Creates function which maps the continuous random variable Uniform([0, 1]) to a discrete uniform random variable
    #     over {0, 1, ..., len(categories) - 1}.
    #     """
    #
    #     def prior(x: float) -> float:
    #         # x=0 has to be handled separately.
    #         # (See https://stackoverflow.com/questions/25688461/ppf0-of-scipys-randint0-2-is-1-0)
    #         if x == 0.0:
    #             return 0.
    #         rv = scipy.stats.randint(low=0, high=len(categories))
    #         # If x is not in [0, 1] (interval includes endpoints), returns np.nan.
    #         # np.nan is not handled here because it is assumed that the input is in the range [0, 1].
    #         index = rv.ppf(x)
    #         return np.int_(index)
    #
    #     return prior
    #
    #
    # @staticmethod
    # def uniform_sampler(min: float, max: float):
    #     rv = UniformPrior(min, max)
    #
    #     def prior(x):
    #         return rv(x)
    #
    #     return prior