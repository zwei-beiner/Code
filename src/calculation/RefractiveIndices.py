# This file provides all refractive indices found on the web, i.e. from Wikipedia in the article 'Sellmeier_equation'
# and the OpenFilter software. The source of the data is given in each function.
# Where it was possible, the refractive index as a function of wavelength was fitted to the Sellmeier equation.
# If this was not possible, because the shape of the refractive index could not be well approximated by the Sellmeier
# equation, or because absorption peaks were in the wavelength range (visible as peaks in the refractive index), the
# data (refrative index vs. wavelength) is stored in a '.csv' file in the 'materials' folder and interpolated.

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path


class Substrates:
    @staticmethod
    def Infrasil301(wl: np.ndarray) -> np.ndarray:
        """
        From: https://www.heraeus.com/media/media/hca/doc_hca/products_and_solutions_8/optics/Data_and_Properties_Optics_fused_silica_EN.pdf

        @param wl: Wavelengths in the range <unknown>.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 4.76523070e-1, 6.27786368e-1, 8.72274404e-1, 2.84888095e-3, 1.18369052e-2, 9.56856012e1
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))


class Materials:
    @staticmethod
    def Nb2O5(wl: np.ndarray) -> np.ndarray:
        """
        From https://refractiveindex.info/?shelf=main&book=Nb2O5&page=Lemarchand, consulted 14/09/2022.

        @param wl: Wavelengths in the range 500nm to 2400nm.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 3.10947718e+00, 8.21924451e-01, 1.60892132e+00, 3.04213563e-02, 7.93014403e-02, -6.40282289e+05
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def SiO2(wl: np.ndarray) -> np.ndarray:
        """
        From: https://en.wikipedia.org/wiki/Sellmeier_equation, consulted 08/09/2022.

        @param wl: Wavelengths in the range <unknown>.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 0.696166300, 0.407942600, 0.897479400, 4.67914826e-3, 1.35120631e-2, 97.9340025
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def MgF2(wl: np.ndarray) -> np.ndarray:
        """
        From: https://en.wikipedia.org/wiki/Sellmeier_equation

        @param wl: Wavelengths in the range <unknown>.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 0.48755108, 0.39875031, 2.3120353, 0.001882178, 0.008951888, 566.13559
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def Ta2O5(wl: np.ndarray) -> np.ndarray:
        """
        Fitted to data from:
        https://refractiveindex.info/?shelf=main&book=Ta2O5&page=Bright-amorphous

        @param wl: Wavelengths in the range 500nm to 2400nm.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 2.95522539, 3.05447357e-01, -1.11866321, 2.80099828e-02, 2.80099739e-02, -3.79621436e+01
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def Ag(wl: np.ndarray) -> np.ndarray:
        """
        Fitted to data from:
        https://refractiveindex.info/?shelf=main&book=Ag&page=Johnson

        @param wl: Wavelengths in the range 500nm to 2400nm.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = -3.56065944e+04, -7.01456109e+02, -1.18909397e+00, 1.68179822e+05, 3.49058377e+05, -7.72053378e-02
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def Al(l: np.ndarray) -> np.ndarray:
        """
        Interpolating data from:
        https://refractiveindex.info/?shelf=main&book=Al&page=Rakic

        @param wl: Wavelengths in the range 500nm to 2400nm.
        @return: Refractive indices at the input wavelengths.
        """
        df = pd.read_csv(Path(__file__).parent / 'materials' / 'Al.csv')
        # Wavelength in nanometers
        wl = df['wl[micrometers]'].values * 1e-6
        # Refractive index
        n = df['n'].values

        f = interp1d(wl, n)

        return f(np.asarray(l))

    @staticmethod
    def Au(wl: np.ndarray) -> np.ndarray:
        """
        Fitted to data from:
        https://refractiveindex.info/?shelf=main&book=Au&page=Werner

        @param wl: Wavelengths in the range 500nm to 2400nm.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 9.24857277e+02, -2.94949743e+06, -2.97284603e+00, -5.24978176e+05, 2.21226071e+06, -5.12980956e-01
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def BK7(wl: np.ndarray) -> np.ndarray:
        """
        From:
        https://en.wikipedia.org/wiki/Sellmeier_equation

        @param wl: Wavelengths in the range <unknown>.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 1.03961212, 0.231792344, 1.01046945, 6.00069867e-3, 2.00179144e-2, 103.560653
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def Sapphire_ordinary_wave(wl: np.ndarray) -> np.ndarray:
        """
        Sapphire for an ordinary wave.

        From:
        https://en.wikipedia.org/wiki/Sellmeier_equation

        @param wl: Wavelengths in the range <unknown>.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 1.43134930, 0.65054713, 5.3414021, 5.2799261e-3, 1.42382647e-2, 325.017834
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def Sapphire_extraordinary_wave(wl: np.ndarray) -> np.ndarray:
        """
        Sapphire for an extraordinary wave.

        From:
        https://en.wikipedia.org/wiki/Sellmeier_equation

        @param wl: Wavelengths in the range <unknown>.
        @return: Refractive indices at the input wavelengths.
        """
        wl = np.asarray(wl)

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 1.5039759, 0.55069141, 6.5927379, 5.48041129e-3, 1.47994281e-2, 402.89514
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def Cr(l: np.ndarray) -> np.ndarray:
        """
        Interpolating data from:
        https://refractiveindex.info/?shelf=main&book=Cr&page=Sytchkova

        @param wl: Wavelengths in the range 500nm to 2400nm.
        @return: Refractive indices at the input wavelengths.
        """
        df = pd.read_csv(Path(__file__).parent / 'materials' / 'Cr.csv')
        # Wavelength in nanometers
        wl = df['wl[micrometers]'].values * 1e-6
        # Refractive index
        n = df['n'].values

        f = interp1d(wl, n)

        return f(np.asarray(l))

    @staticmethod
    def Cu(l: np.ndarray) -> np.ndarray:
        """
        Interpolating data from:
        https://refractiveindex.info/?shelf=main&book=Cu&page=Werner

        @param wl: Wavelengths in the range 500nm to 2400nm.
        @return: Refractive indices at the input wavelengths.
        """
        df = pd.read_csv(Path(__file__).parent / 'materials' / 'Cu.csv')
        # Wavelength in nanometers
        wl = df['wl[micrometers]'].values * 1e-6
        # Refractive index
        n = df['n'].values

        f = interp1d(wl, n)

        return f(np.asarray(l))

    @staticmethod
    def GaAs(l: np.ndarray) -> np.ndarray:
        """
        Interpolating data from:
        https://refractiveindex.info/?shelf=main&book=GaAs&page=Rakic

        @param wl: Wavelengths in the range 500nm to 2400nm.
        @return: Refractive indices at the input wavelengths.
        """
        df = pd.read_csv(Path(__file__).parent / 'materials' / 'GaAs.csv')
        # Wavelength in nanometers
        wl = df['wl[micrometers]'].values * 1e-6
        # Refractive index
        n = df['n'].values

        f = interp1d(wl, n)

        return f(np.asarray(l))

    @staticmethod
    def Ge(l: np.ndarray) -> np.ndarray:
        """
        Interpolating data from:
        https://refractiveindex.info/?shelf=main&book=Ge&page=Nunley

        @param wl: Wavelengths in the range 500nm to 2400nm.
        @return: Refractive indices at the input wavelengths.
        """
        df = pd.read_csv(Path(__file__).parent / 'materials' / 'Ge.csv')
        # Wavelength in nanometers
        wl = df['wl[micrometers]'].values * 1e-6
        # Refractive index
        n = df['n'].values

        f = interp1d(wl, n)

        return f(np.asarray(l))

    @staticmethod
    def Si(wl: np.ndarray) -> np.ndarray:
        """
        Fitted to data from:
        https://refractiveindex.info/?shelf=main&book=Si&page=Schinke

        @param wl: Wavelengths in the range 250nm to 1450nm.
        @return: Refractive indices at the input wavelengths.
        """

        # Sellmeier coefficients. C's are in micrometers^2.
        B1, B2, B3, C1, C2, C3 = 1.05273926e+01, -2.87821661e+05,  2.30593143e+04,  9.76197593e-02, 1.64443249e+06, 2.22825366e+05
        # Wavelength in micrometers
        wl = wl * 1e6
        sq = wl ** 2
        return np.sqrt(1 + B1 * sq / (sq - C1) + B2 * sq / (sq - C2) + B3 * sq / (sq - C3))

    @staticmethod
    def Si3N4(l: np.ndarray) -> np.ndarray:
        """
        Interpolating data from:
        https://refractiveindex.info/?shelf=main&book=Si3N4&page=Luke

        @param wl: Wavelengths in the range 310nm to 5504nm.
        @return: Refractive indices at the input wavelengths.
        """
        df = pd.read_csv(Path(__file__).parent / 'materials' / 'Si3N4.csv')
        # Wavelength in nanometers
        wl = df['wl[micrometers]'].values * 1e-6
        # Refractive index
        n = df['n'].values

        f = interp1d(wl, n)

        return f(np.asarray(l))

    @staticmethod
    def TiO2(l: np.ndarray) -> np.ndarray:
        """
        Interpolating data from:
        https://refractiveindex.info/?shelf=main&book=TiO2&page=Siefke

        @param wl: Wavelengths in the range 310nm to 5504nm.
        @return: Refractive indices at the input wavelengths.
        """
        df = pd.read_csv(Path(__file__).parent / 'materials' / 'TiO2.csv')
        # Wavelength in nanometers
        wl = df['wl[micrometers]'].values * 1e-6
        # Refractive index
        n = df['n'].values

        f = interp1d(wl, n)

        return f(np.asarray(l))


if __name__ == '__main__':
    # Plot all refractive indices.

    import matplotlib.pyplot as plt
    funcs = [Substrates.Infrasil301,
        Materials.Nb2O5, Materials.SiO2, Materials.MgF2, Materials.Ta2O5, Materials.Ag, Materials.Al, Materials.Au,
         Materials.BK7, Materials.Sapphire_ordinary_wave, Materials.Sapphire_extraordinary_wave, Materials.Cr,
         Materials.Cu, Materials.GaAs, Materials.Ge, Materials.Si, Materials.Si3N4, Materials.TiO2
         ]
    titles = ['Infrasil301', 'Nb2O5', 'SiO2', 'MgF2', 'Ta2O5', 'Ag', 'Al', 'Au',
         'BK7', 'Sapphire_ordinary_wave', 'Sapphire_extraordinary_wave', 'Cr',
         'Cu', 'GaAs', 'Ge', 'Si', 'Si3N4', 'TiO2'
         ]
    nrows = int(np.sqrt(len(funcs)))
    ncols = int(np.ceil(len(funcs) / nrows))
    fig: plt.Figure
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3))
    wl = np.linspace(500e-9, 2400e-9, num=100)
    for i, ax in enumerate(fig.get_axes()):
        if i < len(funcs):
            ax: plt.Axes
            ax.plot(wl, funcs[i](wl))
            ax.set_title(titles[i])
        else:
            ax.set_axis_off()
    fig.tight_layout()
    plt.show()
