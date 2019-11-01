# File: proton_flux.py
#
# contains definitions for the differential intensity, rigidity,
# derivative of the rigidity, and the differential flux

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def F(R):
    r"""
    Returns the interstellar differential proton intesity in terms of rigidity

    Parameters
    ----------
    R : float
        rigidity [:math:`\textrm{GV}`]

    Returns
    -------
    F : float
        differential intensity [:math:`\textrm{cm}^{-2}\,\textrm{s}^{-1}\,\textrm{sr}^{-1}\,\textrm{GV}^{-1}`]

    Notes
    -----
    The parameterisation of the flux is given in equation (5) and Table (1) of `1704.06337 <https://arxiv.org/pdf/1704.06337.pdf>`. One additional component that we introduce is a lower bound on the rigidity, :math:`R > 0.2 \, \textrm{GV}` to account for the range of validity of the parametrisation.
    """
    a0, a1, a2, a3, a4, a5 = 94.1, -831.0, 0.0, 16700.0, -10200.0, 0.0
    b, c = 10800.0, 8590.0
    d1, d2 = -4230000.0, 3190.0
    e1, e2, f1, f2, g = 274000.0, 17.4, -39400.0, 0.464, 0.0

    FP1 = a0 + a1*R + a2*np.power(R, 2) + a3*np.power(R, 3) + a4*np.power(R, 4) + a5*np.power(R, 5)
    FP2 = b + c*np.power(R, -1) + d1*np.power(d2 + R, -1) + e1*np.power(e2 + R, -1) + f1*np.power(f2 + R, -1) + g*R

    if R > 1.0:
        return np.power(10.0, -4)*np.power(R, -2.7)*FP2
    elif R >= 0.2:
        return np.power(10.0, -4)*np.power(R, -2.7)*FP1
    else:
        return 0.0

def R(T, A=1.0, Z=1.0, T0=0.9383):
    r"""
    Returns the rigidity for a particle with kinetic energy T, atomic number Z, atomic mass number A, and rest mass :math:`T_0`. The default parameters are for the proton, with rest mass 0.9383 :math:`\textrm{MeV}`.

    Parameters
    ----------
    T : float
        kinetic energy [:math:`\textrm{GeV}`]
    A : float or int
        atomic mass number
    Z : float or int
        atomic number
    T0 : float
        rest mass [:math:`\textrm{GeV}`]

    Returns
    -------
    R : float
        rigidity [:math:`\textrm{GV}`]

    Notes
    -----
    We use the following definition of rigidity,

    .. math:: R(T) = \frac{A}{Z}\sqrt{T^2 + 2 T T_0}

    where :math:`T_0` is the rest mass of the particle.

    Examples
    --------
    >>> R(T=0.1, A=1.0, Z=1.0, T0=0.9383)
    0.44458969848614355
    """
    return (A/Z)*np.power(np.power(T, 2) + 2*T*T0, 0.5)

def dRdT(T, A=1.0, Z=1.0, T0=0.9383):
    r"""
    Returns the derivative of the rigidity for a particle with kinetic energy T, atomic number Z, atomic mass number A, and rest mass :math:`T_0`. The default parameters are for the proton, with rest mass 0.9383 :math:`\textrm{MeV}`.

    Parameters
    ----------
    T : float
        kinetic energy [:math:`\textrm{GeV}`]
    A : float or int
        atomic mass number
    Z : float or int
        atomic number
    T0 : float
        rest mass [:math:`\textrm{GeV}`]

    Returns
    -------
    dRdT : float
        dervative of rigidity [:math:`\textrm{e}^{-1}`]

    Notes
    -----
    The derivative can be calculated analytically from the expression in :py:func:`R`, and is given by,

    .. math:: \frac{\textrm{d}R}{\textrm{d}T} = \frac{A}{Z}\frac{T + T_0}{\sqrt{T^2 + 2 T T_0}}

    Examples
    --------
    >>> dRdT(T=0.1, A=1.0, Z=1.0, T0=0.9383)
    2.3354117370138763
    """
    return (A/Z)*(2*T + 2*T0)*np.power(2*np.power(np.power(T, 2) + 2*T*T0, 0.5), -1.0)

def dPhiPdT(T):
    r"""
    Returns the proton differential flux as a function of the kinetic energy, T. Flux is given in units [:math:`\textrm{cm}^{-2}\,\textrm{s}^{-1}\,\textrm{GeV}^{-1}`].

    Parameters
    ----------
    T : float
        kinetic energy [:math:`\textrm{GeV}`]

    Returns
    -------
    dPhiPdT : float
        differential flux [:math:`\textrm{cm}^{-2}\,\textrm{s}^{-1}\,\textrm{GeV}^{-1}`]

    Notes
    -----
    Simply implements the chain rule :math:`\textrm{d}\phi_p/\textrm{d}T = (\textrm{d}\phi_p/\textrm{d}R)(\textrm{d}R/\textrm{d}T)`.
    """
    return 4*np.pi*F(R(T))*dRdT(T)

def Tmin(Rmin, A=1.0, Z=1.0, T0=0.9383):
    r"""
    Inverts the relation for R(T, A, Z, T0) to find the minimum kinetic energy for which the flux parametrisation is valid.

    Parameters
    ----------
    Rmin : float
        minimum rigidity [:math:`\textrm{GV}`]
    A : float or int
        atomic mass number
    Z : float or int
        atomic number
    T0 : float
        rest mass [:math:`\textrm{GeV}`]

    Returns
    -------
    Tmin : float
        minimum kinetic energy [:math:`\textrm{GeV}`]

    Notes
    -----
    Analytically, this is given by,

    .. math:: T_{\textrm{min}} = -T_0 + \frac{1}{A}\sqrt{(AT_0)^2 + (R_{\textrm{min}}Z)^2}

    Examples
    --------
    >>> Tmin(Rmin=0.2, A=1.0, Z=1.0, T0=0.9383)
    0.021078387290437206
    """
    return -T0 + (1/A)*np.power(np.power(A*T0, 2) + np.power(Rmin*Z, 2), 0.5)

def Tpmin(Tchi, mchi, mp):
    r"""
    Returns the minimum cosmic ray energy required to obtain a dark matter recoil energy Tchi.

    Parameters
    ----------
    Tchi : float
        dark matter kinetic energy [:math:`\textrm{GeV}`]
    mchi : float
        dark matter mass [:math:`\textrm{GeV}`]
    mp : float
        proton mass [:math:`\textrm{GeV}`] (0.9383 :math:`\textrm{GeV}`)

    Returns
    -------
    Tpmin : float
        minimum proton kinetic energy [:math:`\textrm{GeV}`]

    Notes
    -----
    Note that this is a *different* definition of a minimum kinetic energy than in :py:func:`Tmin`. In that case, we are talking about a minimum proton energy arising due to a breakdown in the parametrisation. Here, we are talking about a kinematic constraint. The expression is given in equation (2) of `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`.

    Examples
    --------
    >>> Tpmin(Tchi=0.1, mchi=0.001, mp=0.9383)
    5.812692567523113
    """
    if Tchi > 2*mp:
        return ((Tchi/2) - mp)*(1 + np.power(1 + ((2*Tchi/mchi)*np.power(mp + mchi, 2)/np.power(2*mp - Tchi, 2)), 0.5))
    elif Tchi == 2*mp:
        return 0.0
    else:
        return ((Tchi/2) - mp)*(1 - np.power(1 + ((2*Tchi/mchi)*np.power(mp + mchi, 2)/np.power(2*mp - Tchi, 2)), 0.5))

def Tmax(Tp, mp, mchi):
    r"""
    Maximum kinetic energy that can be transferred to a dark matter particle during collision with the cosmic ray proton.

    Parameters
    ----------
    Tp : float
        kinetic energy of the proton [:math:`\textrm{GeV}`]
    mp : float
        mass of the proton [:math:`\textrm{GeV}`] (0.9383 :math:`\textrm{GeV}`)
    mchi : float
        mass of the dark matter [:math:`\textrm{GeV}`]

    Returns
    -------
    Tmax : float
        the maximum kinetic energy of the dark matter particle [:math:`\textrm{GeV}`]

    Notes
    -----
    The exact expression for this case is given in equation (1) of `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`.

    Examples
    --------
    >>> Tmax(Tp=0.1, mp=0.9383, mchi=0.001)
    0.0004479625471944555
    """
    return (np.power(Tp, 2) + 2*mp*Tp)/(Tp + (np.power(mp + mchi, 2)/(2*mchi)))

def Fp_atmos(R):
    r"""
    Returns the AMS proton flux model in the range :math:`10 \textrm{GV} < R < 1 :math:`\textrm{TV}` from `PRL 114, 171103 (2015) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.114.171103>` using their best fit values for the parameters.

    Parameters
    ----------
    R : float
        rigidity [:math:`\textrm{GeV}`]

    Returns
    -------
    Fp : float
        flux [:math:`\textrm{cm}^{-2}\,\textrm{sr}^{-1}\,\textrm{s}^{-1}\,\textrm{GV}^{-1}`]

    Notes
    -----
    The fit is more complicated than just taking the parameterisation given in the paper. It also interpolates the spectral indices for values of the rigidity below :math:`45\,\textrm{GV}` and fits them to the data contained in the additional materials.
    """
    C, g, dg, s, R0 = 0.4544, -2.849, 0.133, 0.024, 336.0

    R10, C10, n10 = 14.0, 10.87774, -2.521245 # 10.555436, 21.90, -2.521245
    R14, C14, n14 = 19.0, 4.8700, -2.631541 # 14.706936, 9.260, -2.631541
    R19, C19, n19 = 26.0, 2.0856, -2.703690 # 20.308929, 3.894, -2.703690
    R26, C26, n26 = 36.0, 0.8471, -2.768751 # 27.711020, 1.650, -2.768751
    R36, C36, n36 = 45.0, C, -2.790953 # 43.516669, 0.4716, -2.790953

    if 10.0 <= R < 14.0:
        return np.power(10.0, -4)*C10*np.power((R/R10), n10)

    if 14.0 <= R < 19.0:
        return np.power(10.0, -4)*C14*np.power((R/R14), n14)

    if 19.0 <= R < 26.0:
        return np.power(10.0, -4)*C19*np.power((R/R19), n19)

    if 26.0 <= R < 36.0:
        return np.power(10.0, -4)*C26*np.power((R/R26), n26)

    if 36.0 <= R < 45.0:
        return np.power(10.0, -4)*C36*np.power((R/R36), n36)

    if 45.0 <= R <= 1000.0:
        return np.power(10.0, -4)*C*np.power((R/45), g)*np.power(1 + np.power((R/R0), (dg/s)), s)
    else:
        return 0.0

def dPhiPatmosdT(T):
    r"""
    Returns the proton differential flux at the top of the atmosphere as a function of the kinetic energy, T. Flux is given in units [:math:`\textrm{cm}^{-2}\,\textrm{s}^{-1}\,\textrm{GeV}^{-1}`].

    Parameters
    ----------
    T : float
        kinetic energy [:math:`\textrm{GeV}`]

    Returns
    -------
    dPhiPdT : float
        differential flux [:math:`\textrm{cm}^{-2}\,\textrm{s}^{-1}\,\textrm{GeV}^{-1}`]

    Notes
    -----
    In the same way as :py:func:`dPhiPdT`, essentially implements the chain rule using the rigidity definition in :py:func:`dRdT`.
    """
    return 4*np.pi*Fp_atmos(R(T))*dRdT(T)

def Rtilde(Rmin, Rmax):
    r"""Retunrs the correct abscissa based on assuming a flux proportional to :math:`R^{2.7}`.

    Parameters
    ----------
    Rmin : float
        lower rigidity bound [GV]
    Rmax : float
        upper rigidity bound [GV]

    Returns
    -------
    Rtilde : float
        correct abscissa for plotting the binned data point

    Notes
    -----
    Implements the following definition assuming a proton flux proportional to :math:`R^{2.7}`. The rationale for this is contained in `Where to stick your data points? <https://www.sciencedirect.com/science/article/pii/0168900294011125?via%3Dihub>`.

    .. math:: f(\tilde{R}) = \frac{1}{\Delta R} \int_{R_2}^{R_1}{\textrm{d}R\,f(R)}

    Examples
    --------

    >>> Rtilde(Rmin=1.0, Rmax=2.0)
    1.5459611054152882
    """
    return np.power((1/(3.7*(Rmax - Rmin)))*(np.power(Rmax, 3.7) - np.power(Rmin, 3.7)), (1/2.7))

def generate_Rtilde_arr(Rarr):
    r"""
    Generates the :math:`\tilde{R}` array given the array of bin boundaries from the definition in :py:func:`Rtilde`.

    Parameters
    ----------
    Rarr : np.array
        array of bin boundaries [:math:`\textrm{GV}`]

    Returns
    -------
    Rtildearr : np.array
        array of R tilde values [:math:`\textrm{GV}`] for plotting

    Examples
    --------
    >>> generate_Rtilde_arr(Rarr=np.array([10.0, 20.0, 43.0, 56.0]))
    array([15.45961105, 32.65176838, 49.7407944 ])
    """
    Rtildearr = np.array([])
    for idx in range(0, len(Rarr) - 1):
        Rtildearr = np.append(Rtildearr, Rtilde(Rarr[idx], Rarr[idx + 1]))
    return Rtildearr

if __name__ == '__main__':
    plt.style.use('ja')
    plt_color = '#FF2F30'
    hist_color = '#009B72'

    Nbins = 30
    logRmin = np.log10(0.2)
    logRmax = 6.0

    logRarr = np.linspace(logRmin, logRmax, 1000)
    logRbins = np.linspace(logRmin, logRmax, Nbins)
    logReval = logRbins + (logRmax - logRmin)/(2*Nbins)
    Rbins = np.power(10.0, logRbins)
    Reval = np.power(10.0, logReval)
    Rarr = np.power(10.0, logRarr)
    Farr = np.array([])
    Fbins = np.array([])
    for Ri in Rarr:
        Farr = np.append(Farr, F(Ri))
    for Ri in Reval:
        Fbins = np.append(Fbins, F(Ri))

    plt.figure(figsize=(15,8))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.loglog(Rarr, Farr, lw=2.5, c=plt_color)
    #plt.bar(Rbins[:-1], Fbins[:-1], color=hist_color, linewidth=1.0, alpha=0.8, width=1.0*np.diff(Rbins), log=True,ec="k", align="edge")
    ax1.tick_params(labelsize=16)
    ax1.set_title(r'Proton Intensity $\textrm{d}\Phi_p/\textrm{d}R$', fontsize=20)
    ax1.set_xlabel(r'Rigidity $\textrm{[GV]}$', fontsize=16)
    ax1.set_ylabel(r'$\textrm{d}\Phi_p/\textrm{d}R \,\, \textrm{[cm}^{-2}\,\textrm{s}^{-1}\,\textrm{sr}^{-1}\,\textrm{GV}^{-1}\textrm{]}$', fontsize=16)
    #plt.savefig('plots/proton_flux_rigidity.pdf', transparent=True)

    Nbins = 30
    logTmin = np.log10(Tmin(0.2))
    logTmax = 5.0

    logTarr = np.linspace(logTmin, logTmax, 1000)
    logTbins = np.linspace(logTmin, logTmax, Nbins)
    logTeval = logTbins + (logTmax - logTmin)/(2*Nbins)
    Tbins = np.power(10.0, logTbins)
    Teval = np.power(10.0, logTeval)
    Tarr = np.power(10.0, logTarr)
    dPhiParr = np.array([])
    dPhiPbins = np.array([])
    for T in Tarr:
        dPhiParr = np.append(dPhiParr, dPhiPdT(T))
    for T in Teval:
        dPhiPbins = np.append(dPhiPbins, dPhiPdT(T))

    #plt.figure()
    ax2.loglog(Tarr, dPhiParr, lw=2.5, c=plt_color)
    #plt.bar(Tbins[:-1], dPhiPbins[:-1], color=hist_color, linewidth=1.0, alpha=0.8, width=1.0*np.diff(Tbins), log=True,ec="k", align="edge")
    ax2.tick_params(labelsize=16)
    ax2.set_title(r'Differential Flux $\textrm{d}\Phi_p/\textrm{d}T_p$', fontsize=20)
    ax2.set_xlabel(r'Kinetic Energy $\textrm{[GeV]}$', fontsize=16)
    ax2.set_ylabel(r'$\textrm{d}\Phi_p/\textrm{d}T_p \,\, \textrm{[cm}^{-2}\,\textrm{s}^{-1}\,\textrm{GeV}^{-1}\textrm{]}$', fontsize=16)
    plt.savefig('plots/proton_flux_r_and_ke.pdf', transparent=True)
