import numpy as np
from typing import Callable

class MeritFunctionSpecification:
    def __init__(self,
                 p_pol_weighting: float,
                 s_pol_weighting: float,
                 sum_weighting: float,
                 difference_weighting: float,
                 phase_weighting: float,
                 target_reflectivity_s: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros(len(x)),
                 target_reflectivity_p: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros(len(x)),
                 target_sum: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros(len(x)),
                 target_difference: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros(len(x)),
                 target_relative_phase: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros(len(x)),
                 weight_function_s: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones(len(x)),
                 weight_function_p: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones(len(x)),
                 weight_function_sum: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones(len(x)),
                 weight_function_difference: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones(len(x)),
                 weight_function_phase: Callable[[np.ndarray], np.ndarray] = lambda x: np.ones(len(x))):
        self.s_pol_weighting = s_pol_weighting
        self.p_pol_weighting = p_pol_weighting
        self.sum_weighting = sum_weighting
        self.difference_weighting = difference_weighting
        self.phase_weighting = phase_weighting

        self.target_reflectivity_s = target_reflectivity_s
        self.target_reflectivity_p = target_reflectivity_p
        self.target_sum = target_sum
        self.target_difference = target_difference
        self.target_relative_phase = target_relative_phase

        self.weight_function_s = weight_function_s
        self.weight_function_p = weight_function_p
        self.weight_function_sum = weight_function_sum
        self.weight_function_difference = weight_function_difference
        self.weight_function_phase = weight_function_phase
