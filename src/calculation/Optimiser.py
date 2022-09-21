from pathlib import Path
import sys

from BackendCalculations_for_import import BackendCalculations

from typing import Union, Callable

import anesthetic
import anesthetic.kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypolychord
import scipy.stats
import scipy.optimize
from mpi4py import MPI

from Utils import Utils
from WrapperClasses import RefractiveIndex, Wavelength_constraint, n_constraints, d_constraints
from MeritFunctionSpecification import MeritFunctionSpecification


class Optimiser:
    """
    Handles all computations. These are:
    - running the global optimisation using PolyChord (self.run_global_optimisation())
    - running the local optimisation (self.run_local_optimisation())
    - plotting of results (self.make_all_plots())
    - clustering of PolyChord samples (self.do_clustering())
    - decision whether layers can be taken out (self.rerun()) and instantiation of the new Optimiser object for the
      optimisation problem with fewer layers (self.get_new_optimiser())
    """

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
        """
        @param project_name: Name of the project
        @param M: Number of layers
        @param n_outer: Refractive index of the outer medium
        @param n_substrate: Refractive index of the substrate
        @param theta_outer: Incident angle in the outer medium
        @param wavelengths: Wavelengths specification
        @param n_specification: Specification of constraints for the layer refractive indices
        @param d_specification: Specification of constraints for the layer thicknesses
        @param p_pol_weighting: w_{R_p}
        @param s_pol_weighting: w_{R_s}
        @param sum_weighting: w_S
        @param difference_weighting: w_D
        @param phase_weighting: w_{\phi_{sp}}
        @param target_reflectivity_s: \tilde{R}_s
        @param target_reflectivity_p: \tilde{R}_p
        @param target_sum: \tilde{S}
        @param target_difference: \tilde{D}
        @param target_relative_phase: \tilde{\phi}_{sp}
        @param weight_function_s: \delta R_s
        @param weight_function_p: \delta R_p
        @param weight_function_sum: \delta S
        @param weight_function_difference: \delta D
        @param weight_function_phase: \delta \phi_{sp}
        """

        self._project_name = project_name
        # Path to the root directory in which all output is saved. Ensures that if the calculation is rerun with fewer
        # layers, the output results are saved to a different directory.
        self._root: Path = Path.cwd() / self._project_name / f'{M}_layers'

        self._M = M
        self._n_outer = n_outer
        self._n_substrate = n_substrate
        self._theta_outer = theta_outer
        self._wavelengths = Wavelength_constraint(wavelengths)

        # Check if the layer specifications have the correct length.
        if len(n_specification) != self._M or len(d_specification) != self._M:
            raise Exception(f'Invalid specification length: {len(n_specification), len(d_specification), self._M}')
        self._n_constraints = n_constraints(n_specification)
        self._d_constraints = d_constraints(d_specification)

        # Create object which stores all the parameters of the merit function.
        self._merit_function_specification = MeritFunctionSpecification(
            s_pol_weighting=s_pol_weighting,
            p_pol_weighting=p_pol_weighting,
            sum_weighting=sum_weighting,
            difference_weighting=difference_weighting,
            phase_weighting=phase_weighting,
            target_reflectivity_s=target_reflectivity_s,
            target_reflectivity_p=target_reflectivity_p,
            target_sum=target_sum,
            target_difference=target_difference,
            target_relative_phase=target_relative_phase,
            weight_function_s=weight_function_s,
            weight_function_p=weight_function_p,
            weight_function_sum=weight_function_sum,
            weight_function_difference=weight_function_difference,
            weight_function_phase=weight_function_phase
        )

        # Number of parameters to be optimised. This is 'p+q' in the report.
        self._nDims = len(self._n_constraints.get_unfixed_indices()) + len(self._d_constraints.get_unfixed_indices())
        # Index which is used to split parameter array between n (index < split) and d (index > split).
        # This is the integer 'p' in the report.
        self._split = len(self._n_constraints.get_unfixed_indices())

        # Maximum possible thickness, which is the sum of all upper limits on the layer thicknesses. This is passed
        # to the Cython object 'BackendCalculations' which uses it to determine how many sets of wavelengths
        # to pre-allocate.
        D_max: float = self._d_constraints.get_D_max()
        # Minimum and maximum wavelengths in the problem.
        min_wavelength, max_wavelength = self._wavelengths.get_min_max()
        # Prepare refractive index layer specification to pass to the Cython class 'BackendCalculations', which requires
        # a different format: The refractive indices of each layer must be in a list, regardless of whether the
        # constraint if fixed or categorical, i.e. the values of the fixed constraints must be put into a list.
        layer_specification: tuple[list[RefractiveIndex], ...] = tuple(
            val if (type(val) is list) else [val] for _, val in self._n_constraints.get_specification())

        # Create the Cython object 'BackendCalculations' which pre-allocates all arrays for the wavelengths and
        # refractive indices (automatic wavelength calculations, optionally the manually specified wavelengths,
        # arrays of length 1000 for high-resolution plotting). The Cython object also handles all the heavy numerical
        # calculation and provides the calculation of the merit function and the prior function.
        self.backend_calculations = BackendCalculations(
            self._M, self._n_outer, self._n_substrate, self._theta_outer, D_max, min_wavelength, max_wavelength,
            layer_specification, self._merit_function_specification,
            np.array(self._d_constraints.get_fixed_indices(), dtype=np.int_), np.array(self._d_constraints.get_unfixed_indices(), dtype=np.int_),
            np.array(self._d_constraints.get_fixed_values(), dtype=np.float_), self._d_constraints.get_unfixed_values(),
            np.array(self._n_constraints.get_fixed_indices(), dtype=np.int_), np.array(self._n_constraints.get_unfixed_indices(), dtype=np.int_),
            self._n_constraints.get_unfixed_values(),
            self._split, self._nDims, self._wavelengths.get_values() if self._wavelengths.is_fixed() else None
        )

        # PolyChord settings file. Initialised in self.run_global_optimisation() and used again in self.do_clustering().
        self.settings: pypolychord.settings.PolyChordSettings = None


    @property
    def M(self):
        """
        Getter for self._M
        """
        return self._M


    def _build_likelihood_and_prior(self):
        """
        Returns two functions: The likelihood (which is the negative of the merit function) and the prior function
        (which technically is the transformation from the unit hypercube to the parameter space, required by PolyChord
        for sampling).
        """

        # Choose here whether the merit function should be used which uses the manually specified set of wavelengths
        # (then the wavelengths are 'fixed') or whether to use the merit function which uses the automatic wavelength
        # calculation.
        # The choice is made here during initialisation instead of inside the Cython function, because
        # an additional if/else inside the Cython function would make the calculation slower. On the other hand, this
        # means that the two Cython functions 'merit_function_fixed_wavelength' and 'merit_function_auto_wavelength'
        # have almost identical code, which can be dangerous when refactoring the code as the programmer could
        # accidentally forget to refactor the other function after making a change in one function.
        if self._wavelengths.is_fixed():
            merit_function = self.backend_calculations.merit_function_fixed_wavelength
        else:
            merit_function = self.backend_calculations.merit_function_auto_wavelength

        # Create the likelihood function. Since PolyChord is function maximiser, but the merit function must be
        # the negative of the merit function must be passed to PolyChord.
        # The second return value '[]' contains what PolyChord calls the derived parameters, which can be any function
        # of 'params'. Here, we are not interested in any derived parameters so that the list remains empty.
        # Correspondingly, in the function 'self.run_global_optimisation()', 'nDerived=0' must be passed to the
        # PolyChord settings file.
        def likelihood(params):
            return -merit_function(params), []

        return likelihood, self.backend_calculations.prior


    def run_global_optimisation(self) -> None:
        """
        Runs the global optimisation with PolyChord.
        """

        likelihood, prior = self._build_likelihood_and_prior()
        nDerived = 0
        # The number of iterations is set to be 'niter * nDims * nlive' which provides proper scaling of the runtime
        # with the dimensionality of the optimisation problem.
        # Empirically, the number of PolyChord iterations to show convergence of the likelihood function has been
        # determined to be niter=5.
        # Alternatively, one could implement an automatic check if the likelihood has converged by detecting
        # whether a likelihood vs iterations plot flattens off.
        niter = 5

        # Prepare the PolyChord settings file.
        self.settings = pypolychord.settings.PolyChordSettings(self._nDims, nDerived)
        self.settings.nlive = self.settings.nlive
        # If the directory 'polychord_output' exists, it is assumed that a previous run has been interrupted and that
        # the run should be resumed from the already generated data.
        self.settings.read_resume = True if (self._root / 'polychord_output').is_dir() else False

        # At this point, it would be possible to set 'settings.maximise = True', which would cause PolyChord to
        # do a local optimisation after the global optimisation itself, using the Nelder-Mead simplex algorithm. Instead
        # for flexibility and to inspect unfinished runs, a custom local minimiser is used instead in
        # 'self.run_local_optimisation()'.

        # Set the maximum number of iterations (= maximum number of dead points).
        self.settings.max_ndead = int(niter * self._nDims * self.settings.nlive)  # TODO: Check if likelihood converged
        print(f'Maximum number of dead points: {self.settings.max_ndead}')
        sys.stdout.flush()
        # Switch off the default termination criterion of PolyChord, which terminates when the relative change in the
        # evidence Z falls below 'precision_criterion'.
        self.settings.precision_criterion = -1
        # Print the maximum amount of information to stdout for diagnostic purposes.
        self.settings.feedback = 3
        # Set the base directory into which PolyChord writes its files.
        self.settings.base_dir = str(self._root / 'polychord_output')

        # Run PolyChord. This is the most computationally intensive part of the entire program.
        pypolychord.run_polychord(likelihood, self._nDims, nDerived, self.settings, prior)
        print('PolyChord run completed.')
        sys.stdout.flush()


    def calculate_critical_thicknesses(self, optimal_n: np.ndarray):
        """
        With the optimal solution of the refractive indices found by the optimisation, return a two-dimesional numpy
        array containing the critical thicknesses for all layers (axis 0) and all wavelengths (axis 1).
        This is used to determine whether layers can be removed from the multilayer stack.
        """

        # Call a Cython function for heavy computation.
        return self.backend_calculations.calculate_critical_thicknesses(optimal_n)


    def which_layers_can_be_taken_out(self, optimal_n: np.ndarray, optimal_d: np.ndarray) -> list[int]:
        """
        Return a list of indices which are the layers which can be taken out. E.g. [1,4,7] would denote that layers
        1, 4 and 7 can be removed from the multilayer stack.
        @param optimal_n: Optimal refractive indices found by the optimisation
        @param optimal_d: Optimal thicknesses found by the optimisation
        @return:
        """

        # Layer can be taken out for which d < d_crit for all wavelengths.
        d_crit = self.calculate_critical_thicknesses(optimal_n)
        # Compute the minimum across all wavelengths, for each layer.
        d_crit_max = np.amin(d_crit, axis=1)
        return np.argwhere(optimal_d < d_crit_max).flatten().tolist()


    def rerun(self) -> bool:
        """
        Returns True if optimisation should be rerun with fewer layers.
        """

        # Read the optimisation results.
        df = pd.read_csv(self._root / 'optimal_parameters.csv')
        optimal_n: np.ndarray = np.int_(df['n'].values)
        optimal_d: np.ndarray = df['d(nm)'].values * 1e-9

        indices = self.which_layers_can_be_taken_out(optimal_n, optimal_d)
        # Return true if layers should be removed and the new number of layers is at least 1.
        return len(indices) != 0 and (self._M - len(indices)) >= 1


    def plot_critical_thicknesses(self, show_plot: bool, save_plot: bool, root: Path) -> None:
        """
        Create plots for each layer with the critical thickness as a function of wavelength and the optimal thickness
        returned by the optimiser, plotted as a constant value against wavelength.
        This serves as a diagnostic plot to see which layers can be regarded as negligibly thin.
        """

        # Read the optimisation results.
        df = pd.read_csv(root / 'optimal_parameters.csv')
        n: np.ndarray = np.int_(df['n'].values)
        # Calculate d_critical
        d_crit = self.calculate_critical_thicknesses(n)
        # Calculate d_optimal
        d: np.ndarray = df['d(nm)'].values * 1e-9

        # Create the plot.
        fig: plt.Figure
        # Make the grid of subplots approximately square.
        nrows = np.int_(np.floor(np.sqrt(self._M)))
        ncols = int(self._M // nrows + (1 if self._M % nrows != 0 else 0))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        # Iterate through all subplots. Each subplot corresponds to a layer.
        for i, ax in enumerate(fig.axes):
            if i < self._M:
                ax: plt.Axes
                # Get d_critical as a function of wavelength, for this layer.
                d_crit_wavelengths = d_crit[i, :]
                # Create the data for the x-axis.
                wavelengths = np.linspace(*self._wavelengths.get_min_max(), num=len(d_crit_wavelengths))
                ax.plot(wavelengths * 1e9, d_crit_wavelengths * 1e9, label=r'$d_\mathrm{critical}$')
                ax.plot(wavelengths * 1e9, np.full(len(wavelengths), d[i]) * 1e9, label=r'$d_\mathrm{optimal}$')
                ax.set_title(f'Layer {(i + 1)}')
                ax.set_xlabel(r'Wavelength $\lambda$ [nm]')
                ax.set_ylabel('Layer thickness $d$ [nm]')
            else:
                ax.axis('off')
        fig.suptitle('Optimal and critical layer thicknesses against wavelength', fontweight='bold')
        # Leave space at the top for the figure legend.
        fig.tight_layout(rect=[0,0,1,0.94])
        # Set the same legend at the bottom for all subplots.
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 0.94))

        if show_plot:
            plt.show()
        if save_plot:
            fig.savefig(root / 'critical_thicknesses_plot.pdf')

        # Close the figure to deallocate the figure object. Otherwise, during the plotting which occurs in
        # 'self.do_clustering()', the RAM becomes exhausted becayse plt.pyplot keeps all Figure objects which makes
        # it not possible for the garbage collector to deallocate the Figure objects.
        plt.close(fig)
        # return fig, axes


    def _get_max_row_id(self) -> int:
        """
        Returns n_dead from the 'test.resume' file.

        This is important because, if a run is unfinished, PolyChord might not have finished writing all the live
        points, which the 'anesthetic' package then reads as lots of zeroes. To prevent this from happening, the
        last fully saved PolyChord iteration is extracted from the 'test.resume' file (which is the number in
        line 5 of that file) and this number is returned. Later on, this is used to truncate the weighted dataframe
        returned by 'anesthetic' and remove any invalid PolyChord output.
        """
        row_id = -1
        with open(str(self._root / 'polychord_output' / 'test.resume')) as f:
            for i, line in enumerate(f):
                if i == 5:
                    row_id = int(line)
                    break

        if row_id == -1:
            raise ValueError('No n_dead found in test.resume.')

        return row_id


    def _locally_minimise(self, initial_guess: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Runs a 'scipy' local optimiser on the merit function, starting from an initial guess. This method is called
        once after the global optimisation has finished, and is called very often for each cluster found by
        'self.do_clustering()'.

        @param initial_guess: Initial guess to be used for local optimisation.
        @return: The optimal value of 'params' found by the local optimiser and the corresponding merit function value.
        """

        # Get the merit function as the negative of the likelihood. 'scipy' minimises, so that it can be used
        # directly on the merit function (unlike PolyChord, which maximises).
        merit_function = lambda params: -((self._build_likelihood_and_prior()[0])(params)[0])

        # Use bounds so that the local minimiser does not step outside the bounded constraints for the thicknesses
        # or find integers for the refractive indices for which no refractive indices were originally specified.
        # Note: the '+' here is list concatenation.
        # The first list is a list of constraints for each refractive index. Since the scipy minimiser works with
        # floats, but the variables are integers, we set the bounds such that all integers are reachable.
        # For example, if a layer has 3 refractive indices to choose from, the optimal solution should be one of
        # {0, 1, 2}. We use the trick that the minimisation is bounded in the interval (0., 2.9) so that the integers
        # {0, 1, 2} are reachable by the scipy minimiser. Note that the value 0.9 is arbitrary and one could have
        # chosen e.g. 0.3 or 0.77 to the same effect.
        # The bounds on the thicknesses are simply those already specified in the constructor of this 'Optimiser'
        # object.
        bounds = [(0, (len(val) - 1) + 0.9) for val in
                  self._n_constraints.get_unfixed_values()] + self._d_constraints.get_unfixed_values()

        def wrapped_merit_function(params: np.ndarray) -> float:
            split = len(self._n_constraints.get_unfixed_indices())
            # Turn the refractive index input into integers by calling np.floor on them.
            params[:split] = np.floor(params[:split])
            return merit_function(params)

        # Run local optimisation.
        res: scipy.optimize.OptimizeResult = scipy.optimize.minimize(wrapped_merit_function, initial_guess,
                                                                     method='Nelder-Mead', bounds=bounds)
        optimal_params = res.x
        optimal_merit_function_value = res.fun

        # Cast refractive index variables to integers.
        optimal_params[:self._split] = np.floor(optimal_params[:self._split])

        return optimal_params, optimal_merit_function_value


    def _write_to_file(self, optimal_params: np.ndarray, optimal_merit_function_value: float, root: Path) -> None:
        """
        Writes the result of the optimisation to files.

        @param optimal_params: Optimal parameters
        @param optimal_merit_function_value: Optimal merit function value
        @param root: Directory to save files to.
        """
        # Fill positions where the refractive indices are fixed with 0.
        n = np.zeros(self._M, dtype=np.int_)
        n[self._n_constraints.get_fixed_indices()] = 0
        n[self._n_constraints.get_unfixed_indices()] = np.int_(optimal_params[:self._split])

        d = np.zeros(self._M)
        d[self._d_constraints.get_fixed_indices()] = self._d_constraints.get_fixed_values()
        d[self._d_constraints.get_unfixed_indices()] = optimal_params[self._split:]

        indices = self.which_layers_can_be_taken_out(n, d)
        true_or_false: list[bool] = [(i in indices) for i in range(self._M)]

        df = pd.DataFrame({'Layer': np.arange(1, self._M + 1), 'n': n, 'd(nm)': d * 1e9, 'Remove': true_or_false})
        df.to_csv(root / 'optimal_parameters.csv', index=False)

        with open(root / 'optimal_merit_function_value.txt', 'w') as f:
            f.write(str(optimal_merit_function_value))


    def run_local_optimisation(self) -> None:
        """
        Runs the local minimisation after the global minimisation has completed.
        """

        dataframe = anesthetic.NestedSamples(root=str(self._root / 'polychord_output/test'))

        # Restrict dataframe up to where last iteration of PolyChord was performed.
        # This is important in case PolyChord did not finish writing to the file so that remaining lines in the
        # dataframe can be corrupted.
        row_id = self._get_max_row_id()
        dataframe_up_to_max_row = dataframe.iloc[:(row_id + 1), :]

        # Look for the point with the largest value of 'logL' and store the parameters in 'params'.
        max_id = dataframe_up_to_max_row['logL'].idxmax()[0]
        max_row = dataframe_up_to_max_row.iloc[[max_id]]
        params = max_row.loc[:, 0:(self._nDims - 1)].values.flatten()

        optimal_params, optimal_merit_function_value = self._locally_minimise(params)

        # Write to file
        self._write_to_file(optimal_params, optimal_merit_function_value, self._root)


    def plot_merit_function(self, show_plot: bool, save_plot: bool) -> None:
        """
        Plot how the merit function decreases during a PolyChord run.
        More precisely, the merit function value of each dead point is plotted, which is the largest merit function
        value of the samples at a given PolyChord iteration.

        The plot should show a decreasing function which flattens out.
        """

        # Read in PolyChord samples as a weighted data frame (which is a custom class defined in the 'anesthetic'
        # package which inherits from a 'pandas' DataFrame but each row is specified by a pair: the row index and a
        # weight between 0 and 1).
        dataframe = anesthetic.NestedSamples(root=str(self._root / 'polychord_output/test'))

        # Get the last row in the data frame which PolyChord has definitely finished writing to.
        row_id = self._get_max_row_id()

        # Get the merit function values (including the row 'row_id' which is the reason for the '+1')
        merit_function_values = -dataframe['logL'].iloc[:(row_id + 1)].values
        # x-axis is the PolyChord iteration
        x = np.arange(len(merit_function_values))

        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, merit_function_values)
        lower_x_lim, upper_x_lim = ax.get_xlim()
        point_1 = 0, merit_function_values[0]
        point_2 = len(merit_function_values) - 1, merit_function_values[len(merit_function_values) - 1]
        ax.annotate(f'$f={round(point_1[1], 2)}$', xy=point_1, xycoords='data',
                    xytext=(point_1[0] + 0.05 * (upper_x_lim - lower_x_lim), point_1[1]), textcoords='data',
                    arrowprops={'width': 1, 'headwidth': 4, 'headlength': 8, 'facecolor': 'grey'},
                    horizontalalignment='left', verticalalignment='center',
                    )
        ax.annotate(f'$f={round(point_2[1], 2)}$', xy=point_2, xycoords='data',
                    xytext=(point_2[0], point_2[1] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])), textcoords='data',
                    arrowprops={'width': 1, 'headwidth': 4, 'headlength': 8, 'facecolor': 'grey'},
                    horizontalalignment='right', verticalalignment='bottom',
                    )
        ax.set_title('Merit function $f$ against PolyChord iteration', fontweight='bold')
        ax.set_ylabel('Merit function $f$')
        ax.set_xlabel(r'PolyChord iteration (Number of dead points $n_\mathrm{dead}$)')
        fig.tight_layout()

        if show_plot:
            plt.show()
        if save_plot:
            fig.savefig(self._root / 'merit_function_plot.pdf')

        plt.close(fig)
        # return fig, ax


    def plot_reflectivity(self, show_plot: bool, save_plot: bool, root: Path) -> None:
        """
        Plot R_s and R_p as a function of wavelength for the optimal solution.
        Must be run after run_local_optimiser() has finished running.

        @param show_plot: If True, plt.show() is called.
        @param save_plot: If True, the plot is saved as a PDF.
        @return: plt.Figure and plt.Axes objects.
        """

        # Create array containing the optimal solution (indices of which n's to pick and values of d)
        df = pd.read_csv(root / 'optimal_parameters.csv')
        optimal_n = (df['n'].values)[self._n_constraints.get_unfixed_indices()]
        optimal_d = (df['d(nm)'].values * 1e-9)[self._d_constraints.get_unfixed_indices()]

        # Array of wavelengths which will be the x-axis values in the plot.
        wavelengths, res = self.backend_calculations.robustness_analysis(optimal_n, optimal_d)
        # Extract central value (evaluated at optimal_d), and lower and upper error bars.
        means_s, lower_s, upper_s = res[0, :, :]
        means_p, lower_p, upper_p = res[1, :, :]
        means_sum, lower_sum, upper_sum = res[2, :, :]
        means_diff, lower_diff, upper_diff = res[3, :, :]
        means_angle, lower_angle, upper_angle = res[4, :, :]

        # Plot R_s and R_p
        fig1: plt.Figure
        ax_s: plt.Axes
        ax_p: plt.Axes
        fig1, (ax_s, ax_p) = plt.subplots(2, 1, figsize=(9, 6))
        if self._merit_function_specification.s_pol_weighting != 0:
            # Only plot the target if the user has switched on this term in the merit function. That is, do not plot
            # if the user has set self._s_pol_weighting to zero.
            ax_s.plot(wavelengths * 1e9, self._merit_function_specification.target_reflectivity_s(wavelengths), label='Target reflectivity', color='blue', linewidth=0.5)
        ax_s.plot(wavelengths * 1e9, means_s, label='Optimal reflectivity', color='red', linewidth=0.5)
        # Need linewidth=0 as otherwise fill_between leaks colour (See https://github.com/matplotlib/matplotlib/issues/23764).
        ax_s.fill_between(wavelengths * 1e9, lower_s, upper_s, alpha=0.25, color='red', linewidth=0)
        ax_s.set_ylabel(r'$R_\mathrm{s}$')
        ax_s.set_title('Reflectivity against wavelength (s-polarisation)', fontweight='bold')

        if self._merit_function_specification.p_pol_weighting != 0:
            ax_p.plot(wavelengths * 1e9, self._merit_function_specification.target_reflectivity_p(wavelengths), label='Target reflectivity', color='blue', linewidth=0.5)
        ax_p.plot(wavelengths * 1e9, means_p, label='Optimal reflectivity', color='red', linewidth=0.5)
        # Need linewidth=0 as otherwise fill_between leaks colour (See https://github.com/matplotlib/matplotlib/issues/23764).
        ax_p.fill_between(wavelengths * 1e9, lower_p, upper_p, alpha=0.25, color='red', linewidth=0)
        ax_p.set_ylabel(r'$R_\mathrm{p}$')
        ax_p.set_title('Reflectivity against wavelength (p-polarisation)', fontweight='bold')

        # Plot of |R_s+R_p|, |R_s-R_p|, angle_s-angle_p
        fig2: plt.Figure
        ax_sum: plt.Axes
        ax_diff: plt.Axes
        ax_angle: plt.Axes
        fig2, (ax_sum, ax_diff, ax_angle) = plt.subplots(3, 1, figsize=(9, 9))

        if self._merit_function_specification.sum_weighting != 0:
            ax_sum.plot(wavelengths * 1e9, self._merit_function_specification.target_sum(wavelengths), label='Target reflectivity',
                  color='blue', linewidth=0.5)
        ax_sum.plot(wavelengths * 1e9, means_sum, label='Optimal reflectivity', color='red', linewidth=0.5)
        ax_sum.fill_between(wavelengths * 1e9, lower_sum, upper_sum, alpha=0.25, color='red', linewidth=0)
        ax_sum.set_ylabel(r'$|R_\mathrm{s} + R_\mathrm{p}|$')
        ax_sum.set_title('Sum of s- and p-reflectivities against wavelength', fontweight='bold')

        if self._merit_function_specification.difference_weighting != 0:
            ax_diff.plot(wavelengths * 1e9, self._merit_function_specification.target_difference(wavelengths), label='Target reflectivity',
                        color='blue', linewidth=0.5)
        ax_diff.plot(wavelengths * 1e9, means_diff, label='Optimal reflectivity', color='red', linewidth=0.5)
        ax_diff.fill_between(wavelengths * 1e9, lower_diff, upper_diff, alpha=0.25, color='red', linewidth=0)
        ax_diff.set_ylabel(r'$|R_\mathrm{s} - R_\mathrm{p}|$')
        ax_diff.set_title('Difference of s- and p-reflectivities against wavelength', fontweight='bold')

        if self._merit_function_specification.phase_weighting != 0:
            ax_angle.plot(wavelengths * 1e9, self._merit_function_specification.target_relative_phase(wavelengths), label='Target phase difference',
                         color='blue', linewidth=0.5)
        ax_angle.plot(wavelengths * 1e9, means_angle, label='Optimal phase difference', color='red', linewidth=0.5)
        ax_angle.fill_between(wavelengths * 1e9, lower_angle, upper_angle, alpha=0.25, color='red', linewidth=0)
        ax_angle.set_ylabel(r'$\phi_\mathrm{s} - \phi_\mathrm{p}$ [rad]')
        ax_angle.set_title('Phase difference of s- and p-polarised light against wavelength', fontweight='bold')

        for ax in [ax_s, ax_p, ax_sum, ax_diff, ax_angle]:
            ax.set_xlabel(r'Wavelength $\lambda$ [nm]')
            ax.set_xlim(np.amin(wavelengths) * 1e9, np.amax(wavelengths) * 1e9)
            ax.legend()

        for fig in [fig1, fig2]:
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.6)

        if show_plot:
            plt.show()
        if save_plot:
            fig1.savefig(root / 'reflectivity_plot.pdf')
            fig2.savefig(root / 'sum_difference_phase_plot.pdf')

        plt.close(fig1)
        plt.close(fig2)
        # return (fig1, [ax_s, ax_p]), (fig2, [ax_sum, ax_diff, ax_angle])


    def plot_marginal_distributions(self, show_plot: bool, save_plot: bool) -> None:
        """
        Plot the distribution of samples generated during the PolyChord run for each parameter.
        The samples (dead points) are read in and plotted in a grid.
        In a Bayesian context, this would correspond to the marginal distributions of the posterior distribution.

        @param show_plot: Whether to display the plot in a new matplotlib window. If True, plt.show() is called.
        """
        # TODO: This plots the distribution of the dead points. Perhaps plot the live points of the last iteration instead?

        dataframe = anesthetic.NestedSamples(root=str(self._root / 'polychord_output/test'))

        # Using anesthetic package to plot histograms fails for some reason. Possibly because it is trying to
        # draw a smooth curve for a discrete distribution (for the refractive indices).
        # fig, axes = dataframe.plot_1d(list(range(self._nDims)))

        fig: plt.Figure
        axes: list[plt.Axes]
        # Number of rows and columns in the subplot grid. There are self._nDim subplots in total. Try to make the grid
        # square by choosing nrows approximately as the square root self._nDim. Then calculate the number of columns
        # necessary to fit in all the subplots, accounting for the case where self._nDim is not a square number.
        nrows = np.int_(np.floor(np.sqrt(self._nDims)))
        ncols = int(self._nDims // nrows + (1 if self._nDims % nrows != 0 else 0))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        # Loop through all the subplots in the Figure object.
        for i, ax in enumerate(fig.axes):
            ax: plt.Axes
            # Display the y-axis ticks in scientific notation, i.e. 1x10^3 instead of 1000, to save horizontal space.
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # Each subplot corresponds to a free parameter which has been optimised. For i < self._split, the parameter
            # is a refractive index n.
            if i < self._split:
                # Set the title as n_{which layer does this refractive index belong to?}.
                ax.set_xlabel(r'$n_\mathrm{' + str(self._n_constraints.get_unfixed_indices()[i] + 1) + '}$')
                # Get the ith column from the data frame, which are the ith parameter values of the dead points and cast
                # them to integers.
                data = np.int_(dataframe[i].values)
                # Plot a histogram.
                ax.bar(*np.unique(data, return_counts=True))
                # Show only integers on the x-axis.
                ax.set_xticks(np.arange(np.amin(data), np.amax(data) + 1))
            # For i > self._split, the parameter is a thickness d.
            elif i < self._nDims:
                # Set the title as d_{which layer does this thickness belong to?}.
                ax.set_xlabel(r'$d_\mathrm{' + str(self._d_constraints.get_unfixed_indices()[i - self._split] + 1) + '}$ [nm]')
                # Get the ith column from the data frame, which are the ith parameter values of the dead points. Convert
                # the unit 'meter' to 'nanometers'.
                data = dataframe[i].values * 1e9
                # Plot a kernel density estimate (kde) of the probability distribution.
                x, p, _, _ = anesthetic.kde.fastkde_1d(data, np.amin(data), np.amax(data))
                ax.plot(x, p)
            else:
                # Don't show remaining empty plots.
                ax.axis('off')
        fig.suptitle('Distributions of parameter samples after PolyChord run', fontweight='bold')
        fig.tight_layout()

        if show_plot:
            plt.show()
        if save_plot:
            fig.savefig(self._root / 'marginal_distributions_plot.pdf')

        plt.close(fig)
        # return fig, axes


    def make_all_plots(self) -> None:
        """
        Calls all plotting functions and saves plots as PDF files.
        Can only be run after run_local_optimisation() has finished running, because the plotting functions depend on the
        'optimal_parameters.csv' file which is created in run_local_optimisation().
        """

        root = self._root
        self.plot_merit_function(False, True)
        self.plot_marginal_distributions(False, True)
        self.plot_reflectivity(False, True, root)
        self.plot_critical_thicknesses(False, True, root)


    def do_clustering(self) -> None:
        print('Performing cluster analysis on PolyChord samples.')
        sys.stdout.flush()

        from hdbscan import HDBSCAN
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import pairwise_distances
        from scipy.integrate import quad

        # Create distance matrix for n's.
        D = np.empty(self._split, dtype=np.object)
        for k in range(len(D)):
            values = self._n_constraints.get_unfixed_values()[k]
            d = np.zeros((len(values), len(values)), dtype=np.float64)
            # Fill up distance matrix.
            for i in range(len(values)):
                for j in range(i):
                    n_i = values[i]
                    n_j = values[j]
                    def f(x: float) -> float:
                        x = np.array([x])
                        return (n_i(x) - n_j(x)) ** 2
                    d[i, j] = np.sqrt(quad(f, *self._wavelengths.get_min_max())[0] /
                                      (self._wavelengths.get_min_max()[1] - self._wavelengths.get_min_max()[0]))
            # Use symmetry to fill up remaining values, i.e. copy values from d[i, j] into d[j, i]. Use trick of
            # adding the transpose, noting that the unfilled values were initialised to zero.
            d = d + d.T
            # Shift values into the range [0, 1].
            min = np.amin(d)
            max = np.amax(d)
            d = (d - min) / (max - min)
            # Store d in D.
            D[k] = d

        # Define distance metric.
        def dist(theta_1: np.ndarray, theta_2: np.ndarray) -> float:
            n_1 = np.int64(np.floor(theta_1[:self._split]))
            n_2 = np.int64(np.floor(theta_2[:self._split]))
            d_1 = theta_1[self._split:]
            d_2 = theta_2[self._split:]

            res = 0.
            if self._split != 0:
                for i in range(self._split):
                    res += D[i][n_1[i], n_2[i]]
                res /= np.sqrt(self._split)

            res += np.linalg.norm(d_1 - d_2)

            return res


        dataframe = anesthetic.NestedSamples(root=str(self._root / 'polychord_output/test'))
        min_max_scaler = MinMaxScaler()
        # Settings to find as many clusters as possible. Finding more is better than fewer.
        hdbscan = HDBSCAN(min_cluster_size=5, metric='precomputed')
        all_min_points: list[np.ndarray] = []
        all_min_points_logL: list[np.ndarray] = []

        # Total number of iterations at which clustering should be done.
        # Iterations are at 'i * nlive' which corresponds to a parameter space volume compression of 1/e.
        num_iterations = self._get_max_row_id() // self.settings.nlive

        # Split the iterations among nprocs. Each processor does the clustering on a different iteration of PolyChord.
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        subarray = np.array_split(np.arange(num_iterations), nprocs)[rank]

        for i in subarray:
            live_points = dataframe.live_points(i * self.settings.nlive)
            # Get live points at this iteration.
            X = live_points.loc[:, 0:(self._nDims - 1)].values
            logL = live_points['logL'].values

            # Preprocessing d's.
            if not self._d_constraints.all_fixed():
                X[:, self._split:] = min_max_scaler.fit_transform(X[:, self._split:])

            # Distance matrix using custom distance metric.
            m = pairwise_distances(X, metric=dist)
            # Run clustering.
            hdbscan.fit(m)

            # Get cluster labels.
            clusters = np.unique(hdbscan.labels_)
            # Drop outliers.
            clusters = clusters[clusters != -1]

            # Undo the scaling on the thicknesses.
            if not self._d_constraints.all_fixed():
                X[:, self._split:] = min_max_scaler.inverse_transform(X[:, self._split:])

            min_points = np.zeros((len(clusters), self._nDims), dtype=np.float64)
            min_points_logL = np.zeros(len(clusters), dtype=np.float64)
            # In each cluster, find the best solutions.
            for i in clusters:
                # Find points in cluster i.
                in_cluster = (hdbscan.labels_ == i)
                points_in_cluster = X[in_cluster]
                logL_in_cluster = logL[in_cluster]
                # Store point with the maximum logL in cluster i.
                index = np.argmax(logL_in_cluster)
                min_points[i] = points_in_cluster[index]
                min_points_logL[i] = logL_in_cluster[index]

            # Run local minimiser from all collected min_points.
            for i, point in enumerate(min_points):
                # Run local minimiser.
                optimal_params, optimal_merit_function_value = self._locally_minimise(point)
                # Store the solution.
                min_points[i] = optimal_params
                min_points_logL[i] = optimal_merit_function_value

            # Store the points found in this iteration.
            all_min_points.append(min_points)
            all_min_points_logL.append(min_points_logL)

        # Merge all 2D arrays into a single 2D array containing all points.
        optimal_points = comm.allgather(all_min_points)
        # if rank == 0:
        optimal_points = [item for sublist in optimal_points for item in sublist]
        optimal_points = np.vstack(optimal_points)
        # Merge all 1D arrays into a single 1D array.
        optimal_merit_function_values = comm.allgather(all_min_points_logL)
        # if rank == 0:
        optimal_merit_function_values = [item for sublist in optimal_merit_function_values for item in sublist]
        optimal_merit_function_values = np.hstack(optimal_merit_function_values)

        # Remove duplicates. Assuming that the clustering is good and does not produce duplicates, duplicates can
        # arise from detecting the same cluster at different PolyChord iterations.
        def remove_duplicates(X: np.ndarray) -> list[int]:
            """X is a 2D float array."""
            def are_the_same(a: np.ndarray, b: np.ndarray) -> bool:
                return np.array_equal(np.int64(a[:self._split]), np.int64(b[:self._split])) \
                       and np.allclose(a[self._split:], b[self._split:], rtol=1e-02, atol=1e-12)
            indices_to_drop: list[int] = []
            for i in range(len(X)):
                # Skip indices which will be dropped.
                if i not in indices_to_drop:
                    for j in range(i + 1, len(X)):
                        if are_the_same(X[i], X[j]):
                            indices_to_drop.append(j)
            return indices_to_drop
        indices_to_drop = remove_duplicates(optimal_points)
        optimal_points = np.delete(optimal_points, indices_to_drop, axis=0)
        optimal_merit_function_values = np.delete(optimal_merit_function_values, indices_to_drop, axis=0)

        # Sort according to increasing values of logL, i.e. the best solution is at index 0.
        indices = np.argsort(optimal_merit_function_values)
        optimal_points = optimal_points[indices]
        optimal_merit_function_values = optimal_merit_function_values[indices]

        # Split the writing to files among nprocs. Each processor saves some results. The writing of each result to
        # disk is independent of all the others, i.e. it is embarrassingly parallel.
        indices = np.arange(len(optimal_points))
        indices = np.array_split(indices, nprocs)[rank]
        print(f'Found {len(optimal_points)} local minima. Worker {rank}: Saving minima {indices}')
        # Write to files and create plots.
        for i in indices:
            root = self._root / 'local_minima' / f'local_minimum_{i}'
            root.mkdir(parents=True, exist_ok=True)
            self._write_to_file(optimal_points[i], optimal_merit_function_values[i], root)
            self.plot_reflectivity(False, True, root)
            self.plot_critical_thicknesses(False, True, root)


    def get_new_optimiser(self) -> 'Optimiser':
        """
        If self.rerun() is True (i.e. there are layers with thickness below the critical thickness and can be removed),
        this method returns a new Optimiser object with the redundant layers removed.

        @return: Optimiser object with fewer layers.
        """

        # Read in the optimal solution for n and d.
        df = pd.read_csv(self._root / 'optimal_parameters.csv')
        optimal_n: np.ndarray = np.int_(df['n'].values)
        optimal_d: np.ndarray = df['d(nm)'].values * 1e-9

        # Get the indices of the layers for which the thickness is below the critical thickness.
        indices = self.which_layers_can_be_taken_out(optimal_n, optimal_d)
        # Create new specifications for n and d by taking out the layers at the indices.
        new_n_specification = Utils.take_layers_out(self._n_constraints.get_specification(), indices)
        new_d_specification = Utils.take_layers_out(self._d_constraints.get_specification(), indices)

        # Return new Optimiser object with M, n_specification and d_specification modified. All other parameters are
        # kept the same.
        new_optimiser = Optimiser(project_name=self._project_name,
                                  M=self._M - len(indices), # M is reduced by len(indices)
                                  n_outer=self._n_outer,
                                  n_substrate=self._n_substrate,
                                  theta_outer=self._theta_outer,
                                  wavelengths=self._wavelengths.get_values(),
                                  n_specification=new_n_specification, # Use new specification for n
                                  d_specification=new_d_specification, # Use new specification for d
                                  p_pol_weighting=self._merit_function_specification.p_pol_weighting,
                                  s_pol_weighting=self._merit_function_specification.s_pol_weighting,
                                  sum_weighting=self._merit_function_specification.sum_weighting,
                                  difference_weighting=self._merit_function_specification.difference_weighting,
                                  phase_weighting=self._merit_function_specification.phase_weighting,
                                  target_reflectivity_s=self._merit_function_specification.target_reflectivity_s,
                                  target_reflectivity_p=self._merit_function_specification.target_reflectivity_p,
                                  target_sum=self._merit_function_specification.target_sum,
                                  target_difference=self._merit_function_specification.target_difference,
                                  target_relative_phase=self._merit_function_specification.target_relative_phase,
                                  weight_function_s=self._merit_function_specification.weight_function_s,
                                  weight_function_p=self._merit_function_specification.weight_function_p,
                                  weight_function_sum=self._merit_function_specification.weight_function_sum,
                                  weight_function_difference=self._merit_function_specification.weight_function_difference,
                                  weight_function_phase=self._merit_function_specification.weight_function_phase)
        return new_optimiser
