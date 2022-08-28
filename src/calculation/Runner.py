from pathlib import Path

from Optimiser import Optimiser


class Runner:
    def __init__(self, **kwargs):
        self._base_root = Path.cwd() / kwargs['project_name']
        self._optimiser = Optimiser(**kwargs)

    def run(self):
        while True:
            print(f'Running with {self._optimiser.M} layers.')

            files = ['optimal_parameters.csv', 'optimal_merit_function_value.txt', 'merit_function_plot.pdf',
                     'marginal_distributions_plot.pdf', 'reflectivity_plot.pdf', 'critical_thicknesses_plot.pdf']
            root: Path = self._base_root / f'{self._optimiser.M}_layers'

            if not all((root / file).exists() for file in files):
                self._optimiser.run_global_optimisation()
                self._optimiser.run_local_optimisation()
                self._optimiser.make_all_plots()
            if self._optimiser.rerun():
                self._optimiser = self._optimiser.get_new_optimiser()
            else:
                break
