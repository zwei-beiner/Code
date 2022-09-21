from pathlib import Path

import numpy as np


def main():
    from calculation.Compiler import compile_cython_files
    compile_cython_files()

    # Run the calculation
    run()


def run():
    from calculation.Utils import Utils
    from calculation.Runner import Runner
    from calculation.Optimiser import Optimiser

    M = 5
    n_specification = (('fixed', Utils.constant(1.7)), ('categorical', [Utils.constant(0.3), Utils.constant(2.4)]),
                       ('fixed', Utils.constant(0.9)), ('fixed', Utils.constant(0.6)),
                       ('categorical', [Utils.constant(1.3), Utils.constant(5.6), Utils.constant(3.8)]))
    d_specification = (
    ('fixed', 100e-9), ('fixed', 200e-9), ('fixed', 70e-9), ('bounded', (0., 400e-9)), ('bounded', (0., 100e-9)))
    wavelength_specification = (600e-9, 2300e-9)
    # wavelength_specification = np.linspace(600e-9, 2300e-9, num=100)

    # In this problem, there are 6 fixed variables and 4 free variables (2 from n + 2 from d) to be optimised, i.e. nDim=4
    nDim = 4

    # M = 4
    # n_specification = (('fixed', Utils.constant(1.7)), ('categorical', [Utils.constant(0.3), Utils.constant(2.4)]),
    #                    ('fixed', Utils.constant(0.9)), ('categorical', [Utils.constant(1.3), Utils.constant(5.6), Utils.constant(3.8)]))
    # d_specification = (
    #     ('fixed', 100e-9), ('fixed', 200e-9), ('fixed', 70e-9), ('bounded', (0., 100e-9)))
    # wavelength_specification = (600e-9, 2300e-9)
    # # wavelength_specification = np.linspace(600e-9, 2300e-9, num=100)
    #
    # # In this problem, there are 6 fixed variables and 4 free variables (2 from n + 2 from d) to be optimised, i.e. nDim=4
    # nDim = 3

    kwargs = dict(
        project_name='test_project',
        M=M,
        n_outer=Utils.constant(1.00),
        n_substrate=Utils.constant(1.50),
        theta_outer=np.pi / 3,
        wavelengths=wavelength_specification,
        n_specification=n_specification,
        d_specification=d_specification,
        p_pol_weighting=1 / 0.01 ** 2,
        s_pol_weighting=0,
        sum_weighting=0,
        difference_weighting=0,
        phase_weighting=0
    )


    # optimiser = Optimiser(**kwargs)
    # likelihood, prior = optimiser._build_likelihood_and_prior()
    # rng = np.random.default_rng(0)
    # num = int(1e3)
    # times = np.zeros(num)
    # for i in range(num):
    #     tic = time.process_time()
    #     likelihood(prior(rng.uniform(size=nDim)))[0]
    #     toc = time.process_time()
    #     times[i] = toc - tic
    # print(np.mean(times))

    runner = Runner(**kwargs)
    runner.run()

    # optimiser.run_global_optimisation()
    # optimiser.do_clustering()




if __name__ == '__main__':
    main()