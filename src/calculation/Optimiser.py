from pathlib import Path
from typing import Union, Callable

import anesthetic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypolychord
import scipy.stats
import scipy.optimize

import sys

from Utils import Utils
from WrapperClasses import RefractiveIndex, Wavelength_constraint, n_constraints, d_constraints

# Add subdirectory manually to sys.path. This is necessary because we can't place an __init__.py file into
# the subdirectory, as this breaks Cython (this is a known Cython bug)
# This enables us to import modules from the subdirectory directly, e.g. 'import reflectivity_c_file'
file_path = str(Path(__file__).parents[1] / 'calculation')
sys.path.insert(1, file_path)
from reflectivity_c_file import reflectivity, calculate_wavelengths, amplitude



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
        self._root: Path = Path.cwd() / self._project_name / f'{M}_layers'

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

        # Number of parameters to be optimised
        self._nDims = len(self._n_constraints.get_unfixed_indices()) + len(self._d_constraints.get_unfixed_indices())
        # Index which is used to split parameter array between n (index < split) and d (index > split)
        self._split = len(self._n_constraints.get_unfixed_indices())


    @property
    def M(self):
        return self._M


    def _build_amplitude_function(self):
        split = self._split
        nDims = self._nDims

        n_unfixed_values: list[list[RefractiveIndex]] = self._n_constraints.get_unfixed_values()
        n_fixed_values: list[RefractiveIndex] = self._n_constraints.get_fixed_values()
        d_unfixed_values: list[tuple[float]] = self._d_constraints.get_unfixed_values()
        d_fixed_values: list[float] = self._d_constraints.get_fixed_values()

        n_unfixed_indices: list[int] = self._n_constraints.get_unfixed_indices()
        n_fixed_indices: list[int] = self._n_constraints.get_fixed_indices()
        d_unfixed_indices: list[int] = self._d_constraints.get_unfixed_indices()
        d_fixed_indices: list[int] = self._d_constraints.get_fixed_indices()

        # params is an array of length 2M-nFixed where nFixed is the total number of fixed parameters.
        def amplitude_wrapper(params: np.ndarray, wavelength: float, polarisation: int):
            n = np.zeros(self._M)
            n[n_fixed_indices] = [n(wavelength) for n in n_fixed_values]
            n[n_unfixed_indices] = [ns[np.int_(params[:split][i])](wavelength) for i, ns in enumerate(n_unfixed_values)]

            d = np.zeros(self._M)
            d[d_fixed_indices] = d_fixed_values
            d[d_unfixed_indices] = params[split:]

            return amplitude(polarisation, self._M, n, d, wavelength, self._n_outer(wavelength), self._n_substrate(wavelength),
                             self._theta_outer)

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


    def run_global_optimisation(self):
        likelihood, prior = self._build_likelihood_and_prior()
        nDerived = 0
        niter = 5

        settings = pypolychord.settings.PolyChordSettings(self._nDims, nDerived)
        settings.nlive = 10 * settings.nlive
        settings.read_resume = True if (self._root / 'polychord_output').is_dir() else False
        # settings.maximise = True
        settings.max_ndead = int(niter * self._nDims * settings.nlive)  # TODO: Check if likelihood converged
        print(f'Maximum number of dead points: {settings.max_ndead}')
        settings.precision_criterion = -1
        settings.feedback = 3
        settings.base_dir = str(self._root / 'polychord_output')

        pypolychord.run_polychord(likelihood, self._nDims, nDerived, settings, prior)
        print('PolyChord run completed.')


    def calculate_critical_thicknesses(self, optimal_n: list[RefractiveIndex]):
        wavelengths = np.linspace(*self._wavelengths.get_min_max(), num=100)
        d_crit = np.zeros((self._M, len(wavelengths)))
        for i in range(self._M):
            for j in range(len(wavelengths)):
                n_outer = self._n_outer(wavelengths[j])
                n = (optimal_n[i])(wavelengths[j])
                cos_theta = np.sqrt(np.complex_(1 - (n_outer / n) ** 2 * np.sin(self._theta_outer) ** 2))
                k_x = 2 * np.pi/wavelengths[j] * (n / n_outer) * cos_theta
                d_crit[i, j] = np.abs(1 / k_x)
        return 0.01 * d_crit


    def which_layers_can_be_taken_out(self, optimal_n: list[RefractiveIndex], optimal_d: np.ndarray) -> list[int]:
        # Layer can be taken out for which d < d_crit for all wavelengths.
        d_crit = self.calculate_critical_thicknesses(optimal_n)
        d_crit_max = np.amax(d_crit, axis=1)
        return np.argwhere(optimal_d < d_crit_max).flatten().tolist()


    def rerun(self) -> bool:
        """
        Returns True if optimisation should be rerun with fewer layers.
        """
        df = pd.read_csv(self._root / 'optimal_parameters.csv')
        optimal_n: list[RefractiveIndex] = self._n_constraints.get_values_from_indices(np.int_(df['n'].values).tolist())
        optimal_d: np.ndarray = df['d(nm)'].values * 1e-9

        indices = self.which_layers_can_be_taken_out(optimal_n, optimal_d)
        # Return true if layers should be removed and the new number of layers is at least 1.
        return len(indices) != 0 and (self._M - len(indices)) >= 1


    def plot_critical_thicknesses(self, show_plot: bool, save_plot: bool) -> tuple[plt.Figure, list[plt.Axes]]:
        df = pd.read_csv(self._root / 'optimal_parameters.csv')
        n: list[RefractiveIndex] = self._n_constraints.get_values_from_indices(np.int_(df['n'].values).tolist())
        d_crit = self.calculate_critical_thicknesses(n)
        d: np.ndarray = df['d(nm)'].values * 1e-9

        fig: plt.Figure
        nrows = np.int_(np.floor(np.sqrt(self._M)))
        ncols = int(self._M // nrows + (1 if self._M % nrows != 0 else 0))
        fig, axes = plt.subplots(nrows, ncols, figsize=(9, 6))
        for i, ax in enumerate(fig.axes):
            if i < self._M:
                ax: plt.Axes
                d_crit_wavelengths = d_crit[i, :]
                wavelengths = np.linspace(*self._wavelengths.get_min_max(), num=len(d_crit_wavelengths))
                ax.plot(wavelengths * 1e9, d_crit_wavelengths * 1e9, label='$d_\mathrm{critical}$')
                ax.plot(wavelengths * 1e9, np.full(len(wavelengths), d[i]) * 1e9, label='$d_\mathrm{optimal}$')
                ax.set_title(f'Layer {(i + 1)}')
                ax.set_xlabel('Wavelength $\lambda$ [nm]')
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
            fig.savefig(self._root / 'critical_thicknesses_plot.pdf')

        return fig, axes


    def run_local_optimisation(self):
        dataframe = anesthetic.NestedSamples(root=str(self._root / 'polychord_output/test'))

        max_id = dataframe['logL'].idxmax()[0]
        # print(max_id)
        max_row = dataframe.iloc[[max_id]]
        # print(max_row)
        params = max_row.loc[:, 0:(self._nDims - 1)].values.flatten()

        merit_function, _ = self._build_merit_function_and_prior()

        bounds = [(0, (len(val) - 1) + 0.9) for val in self._n_constraints.get_unfixed_values()] + self._d_constraints.get_unfixed_values()
        def wrapped_merit_function(params: np.ndarray) -> float:
            split = len(self._n_constraints.get_unfixed_indices())
            params[:split] = np.floor(params[:split])
            return merit_function(params)
        res: scipy.optimize.OptimizeResult = scipy.optimize.minimize(wrapped_merit_function, params, method='Nelder-Mead', bounds=bounds)
        optimal_params = res.x
        optimal_merit_function_value = res.fun

        # Write to file
        n = np.zeros(self._M, dtype=np.int_)
        n[self._n_constraints.get_fixed_indices()] = 0
        n[self._n_constraints.get_unfixed_indices()] = np.int_(optimal_params[:self._split])

        d = np.zeros(self._M)
        d[self._d_constraints.get_fixed_indices()] = self._d_constraints.get_fixed_values()
        d[self._d_constraints.get_unfixed_indices()] = optimal_params[self._split:]

        optimal_n: list[RefractiveIndex] = self._n_constraints.get_values_from_indices(n.tolist())
        indices = self.which_layers_can_be_taken_out(optimal_n, d.tolist())
        true_or_false: list[bool] = [(i in indices) for i in range(self._M)]

        df = pd.DataFrame({'Layer': np.arange(1, self._M + 1), 'n': n, 'd(nm)': d * 1e9, 'Remove': true_or_false})
        df.to_csv(self._root / 'optimal_parameters.csv', index=False)

        with open(self._root / 'optimal_merit_function_value.txt', 'w') as f:
            f.write(str(optimal_merit_function_value))


    def plot_merit_function(self, show_plot: bool, save_plot: bool) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot how the merit function decreases during a PolyChord run.
        More precisely, the merit function value of each dead point is plotted, which is the largest merit function
        value of the samples at a given PolyChord iteration.

        The plot should show a decreasing function which flattens out.
        """

        dataframe = anesthetic.NestedSamples(root=str(self._root / 'polychord_output/test'))

        merit_function_values = -dataframe['logL'].values
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
        ax.set_xlabel('PolyChord iteration (Number of dead points $n_\mathrm{dead}$)')
        fig.tight_layout()

        if show_plot:
            plt.show()
        if save_plot:
            fig.savefig(self._root / 'merit_function_plot.pdf')

        return fig, ax


    def _robustness_analysis(self, func: Callable[[np.ndarray], np.ndarray], optimal_d: np.ndarray, num_wavelengths):
        """
        @param function: Function which maps array of optimal thicknesses to array of wavelengths
        @return: Array of mean values, array of lower error bars, array of upper error bars
        """
        num_samples = 1000
        rng = np.random.default_rng()
        samples = rng.normal(loc=optimal_d, scale=np.full(shape=len(optimal_d), fill_value=1e-9),
                             size=(num_samples, len(optimal_d)))
        reflectivities = np.zeros((num_samples, num_wavelengths))
        for i in range(num_samples):
            sample = samples[i, :]
            reflectivities[i, :] = func(sample)

        means = np.mean(reflectivities, axis=0)
        stds = np.std(reflectivities, axis=0)
        return means, stds, stds


    def plot_reflectivity(self, show_plot: bool, save_plot: bool) -> tuple[plt.Figure, list[plt.Axes]]:
        df = pd.read_csv(self._root / 'optimal_parameters.csv')
        optimal_params = np.zeros(self._nDims)
        optimal_params[:self._split] = (df['n'].values)[self._n_constraints.get_unfixed_indices()]
        optimal_params[self._split:] = (df['d(nm)'].values * 1e-9)[self._d_constraints.get_unfixed_indices()]

        wavelengths = np.linspace(*self._wavelengths.get_min_max(), num=1000)
        amplitude_wrapper, _ = self._build_amplitude_function()

        reflectivity_s = np.array([np.abs(amplitude_wrapper(optimal_params, wavelength, 0)) ** 2 for wavelength in wavelengths])
        reflectivity_p = np.array([np.abs(amplitude_wrapper(optimal_params, wavelength, 1)) ** 2 for wavelength in wavelengths])

        def calculate_robustness_analysis(polarisation: int):

            def amplitude_new(d: np.ndarray, wavelength: float, polarisation):
                # Calculate the amplitude in the case where all thicknesses are varied.
                # Construct n from optimal values.
                n = np.zeros(self._M)
                n[self._n_constraints.get_fixed_indices()] = [n(wavelength) for n in self._n_constraints.get_fixed_values()]
                n[self._n_constraints.get_unfixed_indices()] = [ns[np.int_(optimal_params[:self._split][i])](wavelength) for i, ns in
                                                                enumerate(self._n_constraints.get_unfixed_values())]
                return amplitude(polarisation, self._M, n, d, wavelength, self._n_outer(wavelength), self._n_substrate(wavelength), self._theta_outer)

            return self._robustness_analysis(lambda d: np.array([np.abs(amplitude_new(d, wavelength, polarisation)) ** 2 for wavelength in wavelengths]), df['d(nm)'].values * 1e-9, len(wavelengths))

        means_s, lower_s, upper_s = calculate_robustness_analysis(0)
        means_p, lower_p, upper_p = calculate_robustness_analysis(1)

        fig: plt.Figure
        ax_s: plt.Axes
        ax_p: plt.Axes
        fig, (ax_s, ax_p) = plt.subplots(2, 1, figsize=(9, 6))
        ax_s.plot(wavelengths * 1e9, self._target_reflectivity_s(wavelengths), label='Target reflectivity', color='blue', linewidth=0.5)
        ax_s.plot(wavelengths * 1e9, reflectivity_s, label='Optimal reflectivity', color='red', linewidth=0.5)
        # Need linewidth=0 as otherwise fill_between leaks colour (See https://github.com/matplotlib/matplotlib/issues/23764).
        ax_s.fill_between(wavelengths * 1e9, means_s - lower_s, means_s + upper_s, alpha=0.25, color='red', linewidth=0)
        ax_s.set_ylabel('$R_\mathrm{s}$')
        ax_s.set_title('Reflectivity against wavelength (s-polarisation)', fontweight='bold')

        ax_p.plot(wavelengths * 1e9, self._target_reflectivity_p(wavelengths), label='Target reflectivity', color='blue', linewidth=0.5)
        ax_p.plot(wavelengths * 1e9, reflectivity_p, label='Optimal reflectivity', color='red', linewidth=0.5)
        # Need linewidth=0 as otherwise fill_between leaks colour (See https://github.com/matplotlib/matplotlib/issues/23764).
        ax_p.fill_between(wavelengths * 1e9, means_p - lower_p, means_p + upper_p, alpha=0.25, color='red', linewidth=0)
        ax_p.set_ylabel('$R_\mathrm{p}$')
        ax_p.set_title('Reflectivity against wavelength (p-polarisation)', fontweight='bold')

        for ax in [ax_s, ax_p]:
            ax.set_xlabel('Wavelength $\lambda$ [nm]')
            ax.set_xlim(np.amin(wavelengths) * 1e9, np.amax(wavelengths) * 1e9)
            ax.legend()

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.6)

        if show_plot:
            plt.show()
        if save_plot:
            fig.savefig(self._root / 'reflectivity_plot.pdf')

        return fig, [ax_s, ax_p]


    def plot_marginal_distributions(self, show_plot: bool, save_plot: bool) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the distribution of samples generated during the PolyChord run for each parameter.
        The samples (dead points) are read in and plotted in a grid.
        In a Bayesian context, this would correspond to the marginal distributions of the posterior distribution.

        @param show_plot: Whether to display the plot in a new matplotlib window. If True, plt.show() is called.
        """


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
        fig, axes = plt.subplots(nrows, ncols, figsize=(9, 6))
        # Loop through all the subplots in the Figure object.
        for i, ax in enumerate(fig.axes):
            ax: plt.Axes
            # Display the y-axis ticks in scientific notation, i.e. 1x10^3 instead of 1000, to save horizontal space.
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # Each subplot corresponds to a free parameter which has been optimised. For i < self._split, the parameter
            # is a refractive index n.
            if i < self._split:
                # Set the title as n_{which layer does this refractive index belong to?}.
                ax.set_xlabel('$n_\mathrm{' + str(self._n_constraints.get_unfixed_indices()[i]) + '}$')
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
                ax.set_xlabel('$d_\mathrm{' + str(self._d_constraints.get_unfixed_indices()[i - self._split]) + '}$ [nm]')
                # Get the ith column from the data frame, which are the ith parameter values of the dead points. Convert
                # the unit 'meter' to 'nanometers'.
                data = dataframe[i].values * 1e9
                # Plot a histogram with the bin widths set automatically because the thickness is a continuous variable.
                ax.hist(data)
            else:
                # Don't show remaining empty plots.
                ax.axis('off')
        fig.suptitle('Histograms of parameter samples after PolyChord run', fontweight='bold')
        fig.tight_layout()

        if show_plot:
            plt.show()
        if save_plot:
            fig.savefig(self._root / 'marginal_distributions_plot.pdf')

        return fig, axes


    def make_all_plots(self) -> None:
        """
        Calls all plotting functions and saves plots as PDF files.
        Can only be run after run_local_optimisation() has finished running, because the plotting functions depend on the
        'optimal_parameters.csv' file which is created in run_local_optimisation().
        """

        self.plot_merit_function(False, True)
        self.plot_marginal_distributions(False, True)
        self.plot_reflectivity(False, True)
        self.plot_critical_thicknesses(False, True)


    def get_new_optimiser(self) -> 'Optimiser':
        """
        If self.rerun() is True (i.e. there are layers with thickness below the critical thickness and can be removed),
        this method returns a new Optimiser object with the redundant layers removed.

        @return: Optimiser object with fewer layers.
        """

        # Read in the optimal solution for n and d.
        df = pd.read_csv(self._root / 'optimal_parameters.csv')
        optimal_n: list[RefractiveIndex] = self._n_constraints.get_values_from_indices(np.int_(df['n'].values).tolist())
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
                                  p_pol_weighting=self._p_pol_weighting,
                                  s_pol_weighting=self._s_pol_weighting,
                                  phase_weighting=self._phase_weighting,
                                  target_reflectivity_s=self._target_reflectivity_s,
                                  target_reflectivity_p=self._target_reflectivity_p,
                                  target_relative_phase=self._target_relative_phase,
                                  weight_function_s=self._weight_function_s,
                                  weight_function_p=self._weight_function_p,
                                  weight_function_phase=self._weight_function_phase)
        return new_optimiser
