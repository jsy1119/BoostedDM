# File: air_density.py
#
# conatins implementation of the polynomial fit for the air density of nitrogen in the atmosphere as a function of height.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
plt.style.use('ja')
plt.rcParams['axes.facecolor'] = 'white'
SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def rho(h):
    r"""
    Returns the density [:math:`\textrm{cm}^{-3}`] of nitrogen atoms in the atmosphere at a height h [:math:`\textrm{km}`]. Data from the `Australian Space academy <https://www.spaceacademy.net.au/watch/debris/atmosmod.htm>`_.

    Parameters
    ----------
    h : float
        height [:math:`\textrm{km}`]

    Returns
    -------
    rho : float
        density [:math:`\textrm{cm}^{-3}`]

    Examples
    --------
    >>> rho(h=180)
    23632408692.397194
    >>> rho(np.linspace(0, 180, 10))
    array([5.05396890e+19, 3.72826592e+18, 1.74036480e+17, 1.45425440e+16,
    7.70753654e+14, 2.34888073e+13, 1.06848906e+12, 1.59714150e+11,
    5.54196208e+10, 2.36324087e+10])
    """
    a0 = 7.001985*np.power(10.0, -2)
    a1 = -4.336216*np.power(10.0, -3)
    a2 = -5.009831*np.power(10.0, -3)
    a3 = 1.621827*np.power(10.0, -4)
    a4 = -2.471283*np.power(10.0, -6)
    a5 = 1.904383*np.power(10.0, -8)
    a6 = -7.189421*np.power(10.0, -11)
    a7 = 1.060067*np.power(10.0, -13)

    logrho = a0 + a1*h + a2*np.power(h, 2) + a3*np.power(h, 3) + a4*np.power(h, 4) + a5*np.power(h, 5) + a6*np.power(h, 6) + a7*np.power(h, 7)

    rho = np.power(10.0, logrho) # kgm^-3

    NA = 6.022*np.power(10.0, 23)
    mN = np.power(10.0, -3)*(14/NA)
    rho = (rho/mN) # [m^-3]
    rho = np.power(10.0, -6)*rho # [cm^-3]
    return rho

def integrate_rho(h, hmax):
    r"""
    In order to calculate the attenuation factor, need to integrate the atmospheric density down to some height h [:math:`\textrm{km}`].

    Parameters
    ----------
    h : float
        height [:math:`\textrm{km}`]
    hmax : float
        height of the atmosphere [:math:`\textrm{km}`]

    Returns
    -------
    integral : float
        integral of the density [:math:`\textrm{cm}^{-2}`]

    Examples
    --------
    >>> integrate_rho(h=100, hmax=180)
    1.4725602921742529e+19
    """
    integral = quad(rho, a=h, b=hmax)[0]
    integral = integral*np.power(10.0, 5)
    return integral

def suppression_factor(h, hmax=180.0, sigmapN=255*np.power(10.0, -27)):
    r"""
    Calculates the flux suppression factor :math:`Y(h)` for the proton flux assuming a constant :math:`pN` cross section across the energy range of interest. Lies in the range :math:`[0, 1]`

    Parameters
    ----------
    h : float
        height [:math:`\textrm{km}`]
    hmax : float
        max height [:math:`\textrm{km}`]
    sigmapN : float
        pN cross section [:math:`\textrm{cm}^2`]

    Returns
    -------
    Y : float
        suppression factor

    Examples
    --------
    >>> suppression_factor(h=40, hmax=180, sigmapN=255*np.power(10.0, -27))
    0.965431731445471
    >>> suppression_factor(h=5, hmax=180, sigmapN=255*np.power(10.0, -27))
    0.001213771406295423

    """
    return np.exp(-sigmapN*integrate_rho(h, hmax))

if __name__ == '__main__':
    plot_dir = '/Users/james/allMyStuff/BoostedDM/Python/plots/'

    hmin, hmax = 0.0, 180.0
    rhoarr = np.array([])
    harr = np.linspace(hmin, hmax, 1000)
    for h in harr:
        rhoarr = np.append(rhoarr, rho(h))


    line_color = '#254441'

    plt.figure()
    plt.title('Air (Nitrogen) Number Density')
    plt.xlabel(r'$h\,\textrm{[km]}$')
    plt.ylabel(r'$n_{^{14}N} \,\, \textrm{[cm}^{-3}\textrm{]}$')
    plt.semilogy(harr, rhoarr, lw=1.0, c=line_color, ls='-')
    axes = plt.axis()
    plt.grid(True)
    plt.axis([hmin, hmax, axes[2], axes[3]])
    plt.savefig('{}air_density.pdf'.format(plot_dir), bbox_inches="tight")
