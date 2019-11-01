# File: pospelov.py
#
# for a given dark matter mass and cross section, computes the
# differential flux due to proton scattering as shown in Figure
# 1 of 1810.10543.

import numpy as np
import matplotlib.pyplot as plt

from proton_flux import dPhiPdT, Tpmin, Tmax

def proton_form_factor(Qsq, L=0.770):
    r"""
    Returns the hadronic elastic scattering form factor as in Eq.(6) of `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`.

    Parameters
    ----------
    Qsq : float
        the kinetic energy transferred (squared) [:math:`\textrm{GeV}^2`]
    L : float
        the reference scale [:math:`\textrm{GeV}`] inversely proportional to the charge radius (for protons, this is 0.770 :math:`\textrm{GeV}`)

    Returns
    -------
    form_factor : float
        the form factor

    Notes
    -----
    The full expression for the form factor is given by,

    .. math:: G_i(Q^2) = \frac{1}{(1 + Q^2/\Lambda_i^2)^2}

    where :math:`\Lambda_i` is the charge radius of the particle in question.

    Examples
    --------
    >>> proton_form_factor(Qsq=np.power(10.0, -4), L=0.770)
    0.9996627603092857
    """
    return np.power(1 + (Qsq/np.power(L, 2.0)), -2.0)

def dPxdTx(Tchi, Deff, rho, mchi, sigma, Tplarge=30.0, npts=1000):
    r"""
    Returns the differential flux for the dark matter by numerically integrating the proton flux in Eq.(8) from 1810.10543

    Parameters
    ----------
    Tchi : float
        kinetic energy of the dark matter [:math:`\textrm{GeV}`]
    Deff : float
        effective distance [:math:`\textrm{cm}`] (0.997 :math:`\textrm{kpc}`)
    rho : float
        energy density of dark matter particles [:math:`\textrm{GeV}\,\textrm{cm}^{-3}`] (0.3 :math:`\textrm{GeV}\,\textrm{cm}^{-3}`)
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]
    sigma : float
        pX cross section [:math:`\textrm{cm}^2`] (:math:`10^{-30}\,\textrm{cm}^2`)
    Tplarge : float
        upper bound for numerical integration [:math:`\textrm{GeV}`]
    npts : int
        number of integration points (1000 default)

    Returns
    -------
    dPxdTx : float
        differential flux of dark matter [:math:`\textrm{GeV}^{-1}\,\textrm{cm}^{-2}\,\textrm{s}^{-1}`]

    Notes
    -----
    The integral being calculated here is given in equation (8) of `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`. We are taking the simplifying assumption that only the protons are contributing. In the full analysis they consider also the effect of Helium.
    """
    prefactor = (Deff*rho*sigma/mchi)*np.power(proton_form_factor(Qsq=2*mchi*Tchi, L=0.770), 2)

    # Numerical Integration

    Tparr = np.linspace(Tpmin(Tchi, mchi, mp=0.9383), Tplarge, npts)
    dTp = (Tparr[-1] - Tparr[0])/npts
    integral = 0.0
    integral -= (0.5*dPhiPdT(Tparr[0]))/Tmax(Tparr[0], mp=0.9383, mchi=mchi) + (0.5*dPhiPdT(Tparr[-1]))/Tmax(Tparr[-1], mp=0.9383, mchi=mchi)
    for Tp in Tparr:
        integral += dPhiPdT(Tp)/Tmax(Tp, mp=0.9383, mchi=mchi)
    integral = integral*dTp
    return prefactor*integral

def plot_pospelov(mchiarr=[0.01, 0.1, 1.0], save_directory='/mnt/james/allMyStuff/BoostedDM/data/'):
    r"""
    On a given plt.figure, plots the Pospelov flux between :math:`10^{-3}` and :math:`1\,\textrm{GeV}`.

    Parameters
    ----------
    mchiarr : np.array
        values of :math:`m_\chi` to plot the curves for
    save_directory : str
        location of the data files containing the pospelov data
    """
    logTchiarr = np.linspace(-3.0, 0.0, 100)
    Tchiarr = np.power(10.0, logTchiarr)
    plotarr = []
    for mchi in mchiarr:
        TdPdTarr = np.load('{}pospelov{}.npy'.format(save_directory, mchi))
        plotarr.append(TdPdTarr)
    idx = 0
    line_colors = ['#D16014', '#E8DB7D', '#355834']
    line_colors = ['#2CEAA3', '#139A43', '#2A6041']
    for TdPdTarr in plotarr:
        plt.loglog(Tchiarr, plotarr[idx], alpha=0.8, lw=1.5, c=line_colors[idx], ls='-', label=r'$m_\chi =$ {} GeV'.format(mchiarr[idx]))
        idx += 1
