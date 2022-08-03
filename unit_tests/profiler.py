import numpy as np
import time

import scipy.stats
import matplotlib.pyplot as plt

from src.main import reflectivity

def profile():
    rng = np.random.default_rng(0)
    num = 3
    for M in range(1, 10):
        print(M)
        for n in (rng.uniform(low=0.1, high=10, size=M) for _ in range(num)):
            for d in (rng.uniform(low=1, high=1e3, size=M) * 1e-9 for _ in range(num)):
                for wavelength in np.linspace(200, 3000, num=num) * 1e-9:
                    for n_outer in np.linspace(0.1, 10, num=num):
                        for n_substrate in np.linspace(0.1, 10, num=num):
                            for theta_outer in np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, num=num):
                                for polarisation in 0, 1:
                                    args = polarisation, M, n, d, wavelength, n_outer, n_substrate, theta_outer
                                    reflectivity(*args)

def execution_time():
    Ms = np.arange(1, 15)
    avg_times = []
    std_times = []

    rng = np.random.default_rng(0)
    num = 4
    for M in Ms:
        print(M)
        times = []

        for n in (rng.uniform(low=0.1, high=10, size=M) for _ in range(num)):
            for d in (rng.uniform(low=1, high=1e3, size=M) * 1e-9 for _ in range(num)):
                for wavelength in np.linspace(200, 3000, num=num) * 1e-9:
                    for n_outer in np.linspace(0.1, 10, num=num):
                        for n_substrate in np.linspace(0.1, 10, num=num):
                            for theta_outer in np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, num=num):
                                for polarisation in 0, 1:
                                    args = polarisation, M, n, d, wavelength, n_outer, n_substrate, theta_outer
                                    tic = time.perf_counter()
                                    reflectivity(*args)
                                    toc = time.perf_counter()
                                    times.append(toc - tic)

        avg_times.append(np.mean(np.array(times)))
        std_times.append(scipy.stats.sem(np.array(times)))

    avg_times = np.array(avg_times)
    std_times = np.array(std_times)

    np.save('execution_times', np.vstack((Ms, avg_times, std_times)))

def plot():
    file = np.load('execution_times.npy')
    Ms, avg_times, std_times = file[0,:], file[1,:], file[2,:]

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    ax.errorbar(Ms, avg_times * 1e3, yerr=std_times * 1e3, fmt='.', capsize=3)
    ax.set_ylabel('Execution time [ms]')
    ax.set_xlabel('Number of layers $M$')
    plt.show()


if __name__ == '__main__':
    profile()
    # execution_time()
    # plot()