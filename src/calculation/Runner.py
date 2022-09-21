from pathlib import Path
import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from Optimiser import Optimiser


class Runner:
    """
    Handles the high-level control loop of the optimisation procedure.
    During the loop iterations, the 'Runner' object keeps an Optimiser object, which handles the details of the
    computation.

    In essence, this class calls the
    (1) global optimisation
    (2) local optimisation on the optimum returned by the global optimisation
    (3) plotting of the results of the global and local optimisations
    (4) clustering of the PolyChord samples created during the global optimisation
    (5) restarting of the next loop iteration with fewer layers.

    Note that steps (2)-(5) are post-processing steps on the results returned by the global optimiser (PolyChord).
    """

    def __init__(self, **kwargs):
        # Path to the directory into which all files will be written.
        self._base_root = Path.cwd() / kwargs['project_name']
        # Initial Optimiser object.
        self._optimiser = Optimiser(**kwargs)

    # def run(self):
    #     while True:
    #         print(f'Running with {self._optimiser.M} layers.')
    #
    #         files = ['optimal_parameters.csv', 'optimal_merit_function_value.txt', 'merit_function_plot.pdf',
    #                  'marginal_distributions_plot.pdf', 'reflectivity_plot.pdf', 'sum_difference_phase_plot.pdf',
    #                  'critical_thicknesses_plot.pdf']
    #         root: Path = self._base_root / f'{self._optimiser.M}_layers'
    #
    #         if not all((root / file).exists() for file in files):
    #             self._optimiser.run_global_optimisation()
    #             self._optimiser.run_local_optimisation()
    #             self._optimiser.make_all_plots()
    #             self._optimiser.do_clustering()
    #         if self._optimiser.rerun():
    #             self._optimiser = self._optimiser.get_new_optimiser()
    #         else:
    #             break

    def run(self):
        # List of files and directories which will be created in each '*_layers' directory.
        # This will be used to check whether the entire optimisation as been completed for a particular
        # number of layers.
        files = ['optimal_parameters.csv', 'optimal_merit_function_value.txt', 'merit_function_plot.pdf',
                 'marginal_distributions_plot.pdf', 'reflectivity_plot.pdf', 'sum_difference_phase_plot.pdf',
                 'critical_thicknesses_plot.pdf']
        directories = ['local_minima']

        # Run the loop until no layers can be removed from the multilayer coating.
        while True:
            if rank == 0:
                # Print only from master to avoid too many prints to the console.
                print(f'Running with {self._optimiser.M} layers.')
                # Force print.
                sys.stdout.flush()

            # Path to the directory into which all files will be writted for the current number of layers in the
            # multilayer coating.
            root: Path = self._base_root / f'{self._optimiser.M}_layers'
            if not (all((root / file).exists() for file in files) and all((root / d).is_dir() for d in directories)):
                # Synchronise processes in advance of PolyChord run.
                comm.barrier()

                # Run global optimisation with PolyChord.
                self._optimiser.run_global_optimisation()

                # Do post-processing on PolyChord output using a single process.
                if rank == 0:
                    self._optimiser.run_local_optimisation()
                    self._optimiser.make_all_plots()

                # Synchronise processes.
                comm.barrier()
                # Clustering uses all processors.
                self._optimiser.do_clustering()
                # Synchronise processes.
                comm.barrier()

            # Let the master process decide whether to run again with fewer layers.
            if rank == 0:
                resume = self._optimiser.rerun()
            else:
                resume = None
            # Broadcast the boolean to all processes.
            resume = comm.bcast(resume, root=0)

            if resume:
                # On all processes, create the new optimiser object using the data written to the file
                # 'optimal_parameters.csv'.
                # Note: An alternative would be to create the new optimiser object on the master process and broadcast
                # it to the other processes. However, an object of the class 'Optimiser' cannot be pickled so that it
                # cannot be sent/received via MPI.
                self._optimiser = self._optimiser.get_new_optimiser()
            else:
                # Else, terminate the loop on all processes.
                break