import numpy as np
import time

import scipy.stats
import matplotlib.pyplot as plt

from src.reflectivity import reflectivity

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

def execution_time(M_min: int, M_max: int, iter: int, concat: bool):
    Ms = np.arange(M_min, M_max)
    avg_times = []
    std_times = []

    rng = np.random.default_rng(0)
    num = iter

    t1 = 0
    t2 = 0
    for M in Ms:
        t2 = time.process_time()
        print(M, f'time to run for this M is: {t2 - t1}')
        t1 = time.process_time()

        times = []

        for n in (rng.uniform(low=0.1, high=10, size=M) for _ in range(num)):
            for d in (rng.uniform(low=1, high=1e3, size=M) * 1e-9 for _ in range(num)):
                for wavelength in np.linspace(200, 3000, num=num) * 1e-9:
                    for n_outer in np.linspace(0.1, 10, num=num):
                        for n_substrate in np.linspace(0.1, 10, num=num):
                            for theta_outer in np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, num=num):
                                for polarisation in 0, 1:
                                    args = polarisation, M, n, d, wavelength, n_outer, n_substrate, theta_outer
                                    tic = time.process_time()
                                    reflectivity(*args)
                                    toc = time.process_time()
                                    times.append(toc - tic)

        avg_times.append(np.mean(np.array(times)))
        std_times.append(scipy.stats.sem(np.array(times)))

    avg_times = np.array(avg_times)
    std_times = np.array(std_times)

    new_data = np.vstack((Ms, avg_times, std_times))
    if concat:
        existing_data = np.load('execution_times.npy')
        new_data = np.concatenate((existing_data, new_data), axis=1)
    np.save('execution_times', new_data)

def plot():
    file = np.load('execution_times.npy')
    Ms, avg_times, std_times = file[0,:], file[1,:], file[2,:]

    res = scipy.stats.linregress(Ms, avg_times)
    s, i = res.slope, res.intercept

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(9,6))
    ax.errorbar(Ms, avg_times * 1e3, yerr=std_times * 1e3, fmt='.', capsize=3)
    ax.plot(Ms, (s*Ms+i)*1e3)
    ax.set_ylabel('Calculation time [ms]')
    ax.set_xlabel('Number of layers $M$')
    textstr = f'slope = {s * 1e3} ms\nintercept = {i * 1e3} ms'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='none',alpha=0.5))
    ax.set_title('Average calculation time of $R(\mathbf{p},\lambda)$ in ms against number of layers $M$')
    plt.show()


if __name__ == '__main__':
    # profile()
    # execution_time(150, 200, 6, True)
    plot()