# File: limits.py


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import scipy.stats

from atmos_dm_flux import kdemultifit
from mean_free_path import TzMin

def get_kernel(particle):
    r"""
    Obtains the kernel and the normalisation factor for the dark matter flux from pions and eta particle decays. This can then be called directly on an array of energies as `kernel.evaluate(energies)`.

    Parameters
    ----------
    particle : str ('pion' or 'eta')
        the particle from which the dark matter particles are produced

    Returns
    -------
    kernel : scipy.stats.gaussian_kde
        the kernel, which can be called directly on an array of energies
    normalisation factor : float
        the normalisation factor for the flux, this is because the scipy.stats.gaussian_kde is normalised by default

    Examples
    --------
    >>> kernel, norm = get_kernel('eta')
    >>> kernel.evaluate(np.power(10.0, -1))
    1.24491518
    >>> norm
    8.905796945913948e-27
    """
    if particle == 'pion':
        save_directory = '/mnt/james/atmos/'
    elif particle == 'eta':
        save_directory = '../data/'
    chi_energies = np.load('{}chienergies.npy'.format(save_directory))
    chi_weights = np.load('{}chiweights.npy'.format(save_directory))
    bw = np.power(10.0, -1.5)
    kernel = scipy.stats.gaussian_kde(dataset=chi_energies, weights=chi_weights, bw_method=bw)
    counts, _ = np.histogram(chi_energies, bins=1, weights=chi_weights)
    norm = counts[0]
    return kernel, norm

def get_Gdet_fit():
    r"""
    Returns the 1d interpolation for the :py:func:`Gdet` function, trained on 100 cross sections between :math:`10^{-34}` and :math:`10^{-28}\,\textrm{cm}^2`.

    Returns
    -------
    GdetFit : scipy.interpolate.interp1d
        Gdet interpolation [:math:`\textrm{cm}^{-2}`]

    Examples
    --------
    >>> GdetFit = get_Gdet_fit()
    >>> GdetFit(np.power(10.0, -34))
    array(2.52933574e+25)
    """
    GdetFit = np.load('GdetFit.npy')[0]
    return GdetFit

def limit_prefactor(mchi=0.001):
    r"""
    Returns the prefactor for the limit plot.

    Parameters
    ----------
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]

    Returns
    -------
    prefactor : float
        the prefactor [:math:`\textrm{s}^{-1}`]

    Notes
    -----
    We are implementing the following definition for the prefactor,

    .. math:: \kappa (\bar{v} \rho_{\textrm{DM}})^{\textrm{ local}}\left(\frac{m_\chi + m_N}{m_\chi + m_p}\right)^2\left(\frac{\sigma^{\textrm{ SI, lim}}_{\textrm{ DM}}}{m_{\textrm{ DM}}}\right)_{m_{\textrm{DM}}\rightarrow \infty}

    In terms of parameter values we are taking :math:`\kappa = 0.23`, :math:`\bar{v} = 250\,\textrm{km}\,\textrm{s}^{-1}`, :math:`\rho_{\textrm{ DM}} = 0.3\,\textrm{GeV}\,\textrm{cm}^{-3}`, :math:`m_N = m_{\textrm{Xe}} = 115.3909 \,\textrm{GeV}`, :math:`\left(\frac{\sigma^{\textrm{ SI, lim}}_{\textrm{ DM}}}{m_{\textrm{ DM}}}\right)_{m_{\textrm{DM}}\rightarrow \infty} = 8.3 \times 10^{-49}\,\textrm{cm}^2 \,\textrm{GeV}^{-1}`, :math:`m_p = 0.938272\,\textrm{GeV}`.

    Examples
    --------
    >>> limit_prefactor(mchi=0.001)
    2.1609020839512918e-32
    """
    k = 0.23
    vbar = 250*np.power(10.0, 5) # cm s^-1
    rhodm = 0.3 # GeV cm^-3
    mXe = 115.3909 # GeV
    mp = 0.938272 # GeV
    sigmamchi = 8.3*np.power(10.0, -49) # cm^2 GeV^-1
    return k*vbar*rhodm*np.power(mchi + mXe, 2.0)*np.power(mchi + mp, -2.0)*sigmamchi

def get_integral(particle, mchi=0.001, dTN=np.power(10.0, -7), dTchi=np.power(10.0, -3)):
    r"""
    Computes the integral in the denominator of equation (16) in `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`_. Computes it in the case of both the pions and the eta fluxes.

    Parameters
    ----------
    particle : str ('pion' or 'dm')
        the dm flux being considered
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]
    dTN : float
        nuclear recoil interval [:math:`\textrm{GeV}`]
    dTchi : float
        dark matter energy interval [:math:`\textrm{GeV}`]

    Returns
    -------
    integral : float
        the result of the double integral [:math:`\textrm{s}^{-1}`]

    Notes
    -----
    We are computing,

    .. math:: \int_{T_1}^{T_2}{\textrm{d}T_N\,\int^{\infty}_{T_\chi(T_\chi^{z, \textrm{min}})}{\textrm{d}T_\chi}\,\frac{1}{T^{\textrm{max}}_{r, N}}\frac{\textrm{d}\phi_\chi}{\textrm{d}T_\chi}  }

    Here :math:`T_1 = 4.9 \,\textrm{keV}` and :math:`40.9\,\textrm{keV}` are the recoil energy bounds in the Xenon 1T detector.
    """
    #TzMin(mchi=mchi, TN=TN)
    TchiMax = np.power(10.0, 1)
    T1 = 4.9*np.power(10.0, -6) # GeV
    T2 = 40.9*np.power(10.0, -6) # GeV
    kernel, norm = get_kernel(particle)
    TNarr = np.arange(T1, T2, dTN)

    integral = 0.0
    count = 1
    print('Starting Integration')
    for TN in TNarr:
        TchiMin = TzMin(mchi=mchi, TN=TN)
        Tchiarr = np.arange(TchiMin, TchiMax, dTchi)
        TrNarr = TrNMax(Tchiarr, mchi)
        fluxarr = norm*kernel.evaluate(Tchiarr)
        farr = fluxarr*np.power(TrNarr, -1.0)
        integral += farr.sum()*dTN*dTchi
        print('Completed {} out of {} recoil energies'.format(count, len(TNarr)), end='\r')
        count += 1
    print('\n--------\nResults:\n--------\n\nIntegral: {}, particle = {}\n'.format(integral, particle))
    return integral

def get_integral_v2(particle, mchi=0.001, dTchi=np.power(10.0, -3.0), TchiMax=np.power(10.0, 1.0)):
    r"""
    Computes the integral in the denominator of equation (16) in `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`_. Computes it in the case of both the pions and the eta fluxes. [version 2]

    Parameters
    ----------
    particle : str ('pion' or 'dm')
        the dm flux being considered
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]
    dTN : float
        nuclear recoil

    Returns
    -------
    integral : float
        the result of the double integral

    Notes
    -----
    We are computing,

    .. math:: \int_{T_1}^{T_2}{\textrm{d}T_N\,\int^{\infty}_{T_\chi(T_\chi^{z, \textrm{min}})}{\textrm{d}T_\chi}\,\frac{1}{T^{\textrm{max}}_{r, N}}\frac{\textrm{d}\phi_\chi}{\textrm{d}T_\chi}  }

    Here :math:`T_1 = 4.9 \,\textrm{keV}` and :math:`40.9\,\textrm{keV}` are the recoil energy bounds in the Xenon 1T detector.
    """
    TchiMin = TzMin(mchi=mchi)
    TchiArr = np.arange(TchiMin, TchiMax, dTchi)
    TNBoundArr = TNBound(TchiArr, mchi=mchi)
    dTNArr = TNBoundArr - 4.9*np.power(10.0, -6.0)
    kernel, norm = get_kernel(particle)
    fluxArr = norm*kernel.evaluate(TchiArr)
    TrNArr = TrNMax(TchiArr, mchi)
    integrands = fluxArr*np.power(TrNArr, -1.0)*dTNArr*dTchi
    integral = integrands.sum()
    print('\n--------\nResults:\n--------\n\nIntegral: {}, particle = {}\n'.format(integral, particle))
    return integral

def TrNMax(Tchi, mchi=0.001):
    r"""
    Returns the maximum recoil energy using equation (1) in `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`_.

    Parameters
    ----------
    Tchi : float
        kinetic energy of the dark matter particle [:math:`\textrm{GeV}`]
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]

    Returns
    -------
    TrNMax : float or np.array
        maximum recoil energy [:math:`\textrm{GeV}`]

    Notes
    -----
    We are implementing the expression,

    .. math:: \frac{T_\chi^2 + 2m_\chi T_\chi}{T_\chi + \frac{(m_\chi + m_{\textrm{Xe}})^2}{2m_\textrm{Xe}}}

    Examples
    --------
    >>> TrNMax(Tchi=np.power(10.0, -2), mchi=0.001)
    2.0794902474678193e-06
    >>> TrNMax(Tchi=np.array([np.power(10.0, -2.0), np.power(10.0, -1.0)]), mchi=0.001)
    array([2.07949025e-06, 1.76481427e-04])
    """
    mXe = 115.3909 # GeV
    num = np.power(Tchi, 2.0) + 2*mchi*Tchi
    denom = Tchi + np.power(mchi + mXe, 2.0)*np.power(2*mXe, -1.0)
    return num*np.power(denom, -1.0)

def get_xenon_rate(gu=5.0*np.power(10.0, -5.0), ms=0.001, dTN=np.power(10.0, -8.0), dTchi=np.power(10.0, -3.0), TchiMax=np.power(10.0, 0)):
    r"""
    Computes the rate for the specific hadrophilic model given in `Batell et al. <http://inspirehep.net/record/1708854>`_ in the Xenon 1T detector. Given an exposure time (278.8 days), the number of expected events can be calculated as :math:`\Gamma_N T_{\textrm{exp}}`. The 90% confidence level is then found by comparing this event number to 3.56 events, from Table I in the most recent `Xenon 1T paper <https://arxiv.org/pdf/1805.12562.pdf>`_.

    Parameters
    ----------
    gu : float
        coupling to up-quark
    ms : float
        mass of the scalar mediator [:math:`\textrm{GeV}`]
    dTN : float
        recoil energy integration resolution [:math:`\textrm{GeV}`]
    dTchi : float
        dark matter energy resolution [:math:`\textrm{GeV}`]
    TchiMax : float
        maximum dark matter energy [:math:`\textrm{GeV}`]

    Returns
    -------
    gN : float
        rate [:math:`\textrm{s}^{-1}`]

    Notes
    -----
    We are implementing the following definition,

    .. math:: \Gamma_N = N_T \int_{T_1}^{T_2}{\textrm{d}T_N \, \int_{T_\chi^{\textrm{min}}(T_N)}^{\infty}{\textrm{d}T_\chi \, \epsilon(T_N)\frac{\textrm{d}\phi_\chi}{\textrm{d}T_\chi}\frac{\textrm{d}\sigma_{\chi N}}{\textrm{d}T_N}  }   }

    Examples
    --------
    >>> get_xenon_rate(gu=5.0*np.power(10.0, -5.0), ms=0.001)
    """
    print('-----------------------------------------------------\nXenon 1T Rate Calculator\n-----------------------------------------------------\nComputing rate for gu = {}, ms = {} GeV'.format(gu, ms))
    print('dTchi = {}, dTN = {}'.format(dTchi, dTN))
    Gdet = 2.529335739713634*np.power(10.0, 25.0) # Assuming Earth is transparent on the lower boundary
    mchi = ms/3
    gchi = 1.0
    A = 131.293
    Z = 54.0
    NT = 1300/(A*1.66054*np.power(10.0, -27))
    epsilon = 0.89
    mN = 0.9314941*A
    mp = 0.938272  # GeV
    mn = 0.939565  # GeV
    mu = 0.0022  # GeV
    yspp = 0.014*gu*mp/mu
    ysnn = 0.012*gu*mn/mu
    br = get_branching_ratio(gu=gu, ms=ms)
    print('\nBranching Ratio = {}'.format(br))
    TchiArr = np.arange(0.015, TchiMax, dTchi)
    TNArr = np.arange(4.9*np.power(10.0, -6.0), 40.9*np.power(10.0, -6.0), dTN)
    TN, TCHI = np.meshgrid(TNArr, TchiArr)
    kernel, norm = get_kernel('eta')
    FluxArr = norm*Gdet*br*kernel.evaluate(TchiArr)
    _, FLUX = np.meshgrid(TNArr, FluxArr)
    prefactor = (np.power(Z*yspp + (A - Z)*ysnn, 2.0)*np.power(gchi, 2.0)/(8*np.pi))
    DSIGMATN = prefactor*(2*mN + TN)*(2*np.power(mchi, 2.0) + mN*TN)*np.power(2*mN*TN + np.power(ms, 2.0), -2.0)*form_factor(TN)
    DSIGMATCHI = np.power(np.power(TCHI, 2.0) + 2*mchi*TCHI, -1.0)
    DSIGMA = DSIGMATN*DSIGMATCHI*np.power(1/(5.06*np.power(10.0, 13)), 2.0)
    INTEGRAND = NT*epsilon*DSIGMA*FLUX*np.heaviside(TNBound(TCHI) - TN, 1.0)
    gN = INTEGRAND.sum()*dTchi*dTN
    events = gN*278.8*86400
    print('\n--------\nResults:\n--------\n\nRate: {} per second, Events = {}\n'.format(gN, events))
    return gN, events


def form_factor(TN):
    r"""
    Returns the Helm Form Factor from `0608.035 <https://arxiv.org/pdf/hep-ph/0608035.pdf>`_.

    Parameters
    ----------
    TN : np.array
        nuclear recoil energy [:math:`\textrm{GeV}`]

    Returns
    -------
    FHsq : np.array
        square of the Helm Form Factor

    Examples
    --------
    >>> form_factor(TN=5.0*np.power(10.0, -6.0))
    0.7854737713313549
    """
    A = 131.293
    mN = 0.9314941*A
    q = np.sqrt(2*mN*TN)
    fmGeV = 1/(1.98*np.power(10.0, -1.0))
    s = 0.9*fmGeV
    a = 0.52*fmGeV
    c = (1.23*np.power(A, 1/3) - 0.60)*fmGeV
    R1 = np.sqrt(np.power(c, 2.0) + (7/3)*np.power(np.pi, 2.0)*np.power(a, 2.0) - 5*np.power(s, 2.0))
    FHsq = 9*np.exp(-np.power(q*s, 2.0))*np.power(q*R1, -2.0)*np.power(np.sin(q*R1)*np.power(q*R1, -2.0) - np.cos(q*R1)*np.power(q*R1, -1.0), 2.0)
    return FHsq

def get_branching_ratio(gu, ms):
    r"""
    Returns the branching ratio for the process :math:`\eta \rightarrow \pi^0 \chi\chi` from the model in `Batell et al. <https://arxiv.org/pdf/1812.05103.pdf>`_.

    Parameters
    ----------
    gu : float
        coupling to up-quark
    ms : float
        scalar mediator mass [:math:`\textrm{GeV}`]

    Returns
    -------
    br : float
        branching ratio :math:`\textrm{BR}(\eta \rightarrow \pi^0 \chi\chi)`

    Notes
    -----
    We are implementing the following expression,

    .. math:: \textrm(\eta \rightarrow \pi^0 S) = \frac{C^2 g_u^2 B^2}{16\pi m_\eta \Gamma_\eta}\lambda^{1/2}\left(1, \frac{m_S^2}{m_\eta^2}, \frac{m_\pi^2}{m_\eta^2}\right)

    where we are assuming a branching ratio :math:`\textrm{BR}(S \rightarrow \chi\chi) = 1`, and :math:`\lambda(a, b, c) = a^2 + b^2 + c^2 - 2(ab + bc + ac)`.

    Examples
    --------
    >>> get_branching_ratio(gu=7*np.power(10.0, -4.0), ms=0.1)
    0.05765234404584737
    """
    thetapr = -np.pi/9
    mpi = 0.1349770 # GeV
    meta = 0.547862 # GeV
    mu = 0.0022 # GeV
    md = 0.0047 # GeV
    geta = 1.31*np.power(10.0, -6.0) # GeV
    C = np.sqrt(1/3)*np.cos(thetapr) - np.sqrt(2/3)*np.sin(thetapr)
    B = np.power(mpi, 2.0)*np.power(mu + md, -1.0)
    a = 1.0
    b = np.power(ms/meta, 2.0)
    c = np.power(mpi/meta, 2.0)
    br = np.power(C*gu*B, 2.0)*np.power(16*np.pi*meta*geta, -1.0)*np.sqrt(np.power(a, 2) + np.power(b, 2) + np.power(c, 2) - 2*a*b - 2*b*c - 2*a*c)
    return br


def TNBound(Tchi, mchi=0.001, mN=115.3909, Trecoil_max=40.9*np.power(10.0, -6.0)):
    r"""
    Inverts the relation for the expression in :py:func:`TzMin` to find the maximum recoil energy that can be generated from a dark matter kinetic energy :math:`T_\chi`.

    Parameters
    ----------
    Tchi : float
        kinetic energies of dark matter particle [:math:`\textrm{GeV}`]
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]
    mN : float
        mass of the recoil nucleus [:math:`\textrm{GeV}`]
    Trecoil_max : float
        maximum detector sensitivity for recoil energies [:math:`\textrm{GeV}`]

    Returns
    -------
    TNBound : np.array
        maximum recoil energy, either kinematically or by detector sensitivity

    Notes
    -----
    We are implementing the following,

    .. math:: T_N^{\textrm{bound}} = \textrm{min}\left[\frac{2(2 m_\chi + T_\chi)m_N T_\chi}{m_\chi^2 + 2m_\chi m_N + m_N^2 + 2m_N T_\chi}, T_N^{\textrm{recoil}}\right]

    For Xenon 1T, :math:`T_N^{\textrm{recoil}} = 40.9\,\textrm{keV}` and :math:`m_N = 115.3909\,\textrm{GeV}`.

    Examples
    --------
    >>> TNBound(np.array([0.015, 0.020]), mchi=0.001, mN=115.3909, Trecoil_max=40.9*np.power(10.0, -6.0))
    array([4.41853393e-06, 7.62347650e-06])
    >>> TNBound(np.array([0.040, 0.050, 0.055]), mchi=0.001, mN=115.3909, Trecoil_max=40.9*np.power(10.0, -6.0))
    array([2.90977363e-05, 4.09000000e-05, 4.09000000e-05])
    """
    TNBound = 2*mN*Tchi*(2*mchi + Tchi)*np.power(np.power(mchi, 2.0) + 2*mchi*mN + np.power(mN, 2.0) + 2*mN*Tchi, -1.0)
    mask = (TNBound > Trecoil_max)
    TNBound[mask] = Trecoil_max
    return TNBound

def get_miniboone_data():
    r"""
    Loads miniboone data from miniboone.csv. Data taken from `Batell et al. <http://inspirehep.net/record/1708854>`_.

    Returns
    -------
    mini_df : pd.DataFrame
        miniboone dataframe with columns [:math:`m_S` (in GeV), :math:`g_u`]
    """
    mini_df = pd.read_csv('miniboone.csv', header=None)
    mini_df.columns = ['ms [GeV]', 'gu']
    return mini_df

if __name__ == '__main__':
    # dTchiArr = np.logspace(-3.0, -2.0)
    # eventsarr = np.empty(len(dTchiArr))
    # idx = 0
    # for dTchi in dTchiArr:
    #     rate, events = get_xenon_rate(dTchi=dTchi)
    #     eventsarr[idx] = events
    #     idx += 1
    # plt.figure()
    # plt.semilogx(dTchiArr, eventsarr)
    # plt.title(r'Rate Integral Convergence')
    # plt.xlabel(r'$\Delta T_\chi\,\textrm{[GeV]}$')
    # plt.ylabel(r'Events')
    # plt.savefig('plots/events.pdf')

    # HADROPHILIC PLOT
    import matplotlib
    plt.rcParams['axes.linewidth'] = 1.75
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    guArr = np.logspace(-5.0, -3.0, 25)
    msArr = np.logspace(-3.0, np.log10(0.5), 25)
    MS, GU = np.meshgrid(msArr, guArr)
    # EVENTS = np.empty([len(guArr), len(msArr)])
    # idx = 1
    # for i in range(0, len(guArr)):
    #     for j in range(0, len(msArr)):
    #         print('Completed {} out of {}'.format(idx, len(msArr)*len(guArr)))
    #         rate, events = get_xenon_rate(ms=msArr[j], gu=guArr[i])
    #         EVENTS[i][j] = events
    #         idx += 1
    # np.save('events.npy', EVENTS)
    EVENTS = np.load('events.npy')
    LZ_EVENTS = (1.0/0.89)*((5.6*1000)/(1.3*278.8))*EVENTS
    #print(EVENTS.shape)
    event_lim = 3.56
    cmap = 'Greens'
    from matplotlib import ticker
    miniboone_df = get_miniboone_data()
    plt.figure()
    plt.plot(miniboone_df['ms [GeV]'], miniboone_df['gu'], color='#62656E', lw=1.0)
    #plt.contourf(MS, GU, EVENTS, locator=ticker.LogLocator(), cmap=cmap, alpha=1.0)
    #plt.colorbar(label='Event Count')
    CS = plt.contour(MS, GU, EVENTS, levels=[event_lim], colors=['#D81159'], linewidths=1.0, alpha=1.0, linestyles='-')
    plt.contour(MS, GU, LZ_EVENTS, levels=[event_lim], colors=['#D81159'], linewidths=1.0, alpha=1.0, linestyles='--')
    mss, guu = CS.allsegs[0][0].T
    plt.fill(np.append(miniboone_df['ms [GeV]'], mss[::-1]), np.append(miniboone_df['gu'], guu[::-1]), edgecolor='k', facecolor='#62656E', alpha=0.3, linewidth=0.0)
    mss = np.append(mss, np.array([mss[-1], mss[0]]))
    guu = np.append(guu, np.array([np.power(10.0, -3), np.power(10.0, -3)]))
    plt.fill(mss, guu, edgecolor='k', facecolor='#D81159', alpha=0.3, linewidth=0.0)
    plt.xlabel(r'$m_S\,\mathrm{[GeV]}$')
    plt.ylabel(r'$g_u$')
    #plt.title('Hadrophilic Exclusion Limits')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    axes = plt.axis()
    plt.axis([np.power(10.0, -3.0), mss.max(), 5*np.power(10.0, -6.0), np.power(10.0, -3.0)])
    #plt.scatter(MS, GU, c='k', marker='x', s=4.0)
    axes = plt.axis()
    plt.text(2*np.power(10.0, -2.0), 1.05*miniboone_df['gu'][0], 'MINIBOONE', fontsize=12, color='#62656E')
    plt.text(1.1*axes[0], 5.5*np.power(10.0, -5.0), r'ICRDM ($\eta$) (XENON1T)', fontsize=12, color='#D81159')
    plt.text(1.1*axes[0], 2.55*np.power(10.0, -5.0), r'ICRDM ($\eta$) (LZ)', fontsize=12, color='#D81159')

    plt.text(np.power(10.0, -2.9), np.power(10.0, -5.1), r'$g_\chi = 1$')
    plt.text(np.power(10.0, -2.9), np.power(10.0, -5.25), r'$m_\chi = m_S/3$')
    xmin, xmax = axes[0], axes[1]
    ymin, ymax = axes[2], axes[3]
    xy = (xmin, ymin)
    width = xmax - xmin
    height = ymax - ymin
    import matplotlib.patches as patches
    # create the patch and place it in the back of countourf (zorder!)
    p = patches.Rectangle(xy, width, height, hatch=4*'/', fill=None, alpha=0.2, zorder=-10)
    ax = plt.gca()
    ax.tick_params(which='minor', length=4)
    #ax.add_patch(p)
    plt.savefig('plots/hadrophilic.pdf')


    # dTchi = np.power(10.0, -4)
    # dtc = 'minus4'
    # TchiMaxArr = np.logspace(-1.0, 0.0, 25)
    # integrals = np.empty(len(TchiMaxArr))
    # idx = 0
    # for TchiMax in TchiMaxArr:
    #     integrals[idx] = get_integral_v2('eta', mchi=0.001, TchiMax=TchiMax, dTchi=dTchi)
    #     idx += 1
    # np.save('TchiMax{}.npy'.format(dtc), TchiMaxArr)
    # np.save('integrals{}.npy'.format(dtc), integrals)

    # print(integrals)
    # plt.semilogx(TchiMaxArr, integrals)
    # plt.show()
    # plt.figure()
    # # dtc = 'minus1'
    # # integrals = np.load('integrals{}.npy'.format(dtc))
    # # TchiMaxArr = np.load('TchiMax{}.npy'.format(dtc))
    # # plt.semilogx(TchiMaxArr, integrals/np.power(10.0, -28.0), label = r'$10^{-1}$')
    # dtc = 'minus2'
    # integrals = np.load('integrals{}.npy'.format(dtc))
    # TchiMaxArr = np.load('TchiMax{}.npy'.format(dtc))
    # plt.semilogx(TchiMaxArr, integrals/np.power(10.0, -28.0), label = r'$10^{-2}$')
    # dtc = 'minus3'
    # integrals = np.load('integrals{}.npy'.format(dtc))
    # TchiMaxArr = np.load('TchiMax{}.npy'.format(dtc))
    # plt.semilogx(TchiMaxArr, integrals/np.power(10.0, -28.0), label = r'$10^{-3}$')
    # dtc = 'minus4'
    # integrals = np.load('integrals{}.npy'.format(dtc))
    # TchiMaxArr = np.load('TchiMax{}.npy'.format(dtc))
    # plt.semilogx(TchiMaxArr, integrals/np.power(10.0, -28.0), label = r'$10^{-4}$')
    # plt.title('Flux Integral Convergence')
    # plt.xlabel(r'$T_\chi^{\textrm{\small max}}\,\textrm{[GeV]}$')
    # plt.ylabel(r'Integral $\textrm{[}10^{-28}\,s^{-1}\textrm{]}$')
    # plt.legend(loc='lower right', fontsize=10, title=r'$\Delta T_\chi$', title_fontsize=10)
    # plt.savefig('plots/int_convergence.pdf')


    from matplotlib import ticker
    from scipy.interpolate import interp1d
    prefactor = limit_prefactor()
    integral = 1.871660096591386*np.power(10.0, -26)
    integral = 2.061418762860559*np.power(10.0, -27)
    integral = 5.863077015127929*np.power(10.0, -28)
    eta_integral = 5.863064797490202*np.power(10.0, -28.0) # get_integral_v2('eta')
    pion_integral = 7.190987691388251*np.power(10.0, -27.0) # get_integral_v2('pion')
    eta_level = prefactor*np.power(eta_integral, -1.0)
    pion_level = prefactor*np.power(pion_integral, -1.0)
    LZ_FACTOR = (3.3217544185377643e-47/1757.9236139586912)/(8.3*np.power(10.0, -49))

    pion_color = '#01295F'
    pion_color = '#29AB87'
    eta_color = '#D81159'
    pospelov_color = '#FF8514'


    # size = 100
    # sigmaarray = np.logspace(-34.0, -28.0, size)
    # Gdetarr = np.load('Gdetarr.npy')
    # sigmaarray = np.append(np.logspace(-40.0, -34.0, size)[:-1], sigmaarray)
    # Gdetarr = np.append(np.full(size - 1, Gdetarr[0]), Gdetarr)
    #
    # GdetFun = interp1d(sigmaarray, Gdetarr, kind='slinear')


    # GdetFun = get_Gdet_fit()
    # B = np.array([np.power(10.0, -2.0), np.power(10.0, -4.0), np.power(10.0, -6.0), np.power(10.0, -7.0), np.power(10.0, -8.0)])
    # labels = np.array([r'$10^{-2}$', r'$10^{-4}$', r'$10^{-6}$', r'$10^{-7}$', r'$10^{-8}$'])
    # S = np.logspace(-35.0, -28.0)
    # idx = 0
    # for br in B:
    #     plt.loglog(S, S*GdetFun(S)*br, label=labels[idx])
    #     idx += 1
    # plt.plot([S[0], S[-1]], [eta_level, eta_level], 'k--')
    # plt.plot([S[0], S[-1]], [pion_level, pion_level], 'k--')
    # plt.text(np.power(10.0, -34.8), np.power(10.0, -10.3), 'Eta Flux', fontsize=8)
    # plt.text(np.power(10.0, -34.2), np.power(10.0, -11.4), 'Pion Flux', fontsize=8)
    # plt.title('Limits for different branching ratios')
    # plt.legend(fontsize=9, title=r'$\textrm{BR}(\eta \rightarrow \chi\chi)$', title_fontsize=9, frameon=True, fancybox=True)
    # plt.xlabel(r'$\sigma_\chi^{\textrm{\tiny SI}}\,\textrm{[cm}^2\textrm{]}$')
    # plt.ylabel(r'$\sigma_\chi^{\textrm{\tiny SI}} G(\sigma_\chi^{\textrm{\tiny SI}})\textrm{BR}(\eta \rightarrow \chi\chi)$')
    # plt.savefig('plots/gdetbrs.pdf')


    GdetFun = get_Gdet_fit()
    lw = 1.0
    alpha = 0.3
    pospelov_top = 2.411345e-28
    pospelov_low = 1.543418e-31

    plt.figure()
    plt.plot([np.power(10.0, -10.0), np.power(10.0, -1.0)], [pospelov_top, pospelov_top], c=pospelov_color, label=r'Elastic CRDM', lw=lw)
    plt.plot([np.power(10.0, -10.0), np.power(10.0, -1.0)], [pospelov_low, pospelov_low], c=pospelov_color, lw=lw)
    plt.fill([np.power(10.0, -10.0), np.power(10.0, -1.0), np.power(10.0, -1.0), np.power(10.0, -10.0)], [pospelov_low, pospelov_low, pospelov_top, pospelov_top], edgecolor=pospelov_color, facecolor=pospelov_color, alpha=alpha, linewidth=0.0), #fill=False, hatch=2*'//')
    B, S = np.meshgrid(np.logspace(-9.0, np.log10(5*10**(-2)), 10000), np.logspace(-38.0, -28.0, 10000))
    C = S*GdetFun(S)*B
    CS = plt.contour(B, S, C, levels=[eta_level], colors=[eta_color], linewidths=0.0)
    plt.contour(B, S, C, levels=[eta_level*LZ_FACTOR], colors=[eta_color], linewidths=1.0, linestyles='--')
    br, sig = CS.allsegs[0][0].T
    plt.fill(br, sig, edgecolor='k', facecolor=eta_color, alpha=alpha, linewidth=0.0)

    plt.plot(br, sig, c=eta_color, lw=lw, label=r'Inelastic CRDM ($\eta$)')
    #plt.plot([br[1], br[-1]], [sig[1], sig[-1]], c=eta_color, lw=1.0)

    B, S = np.meshgrid(np.logspace(-10.0, np.log10(6*10**(-4)), 10000), np.logspace(-38.0, -28.0, 10000))
    C = S*GdetFun(S)*B
    CS = plt.contour(B, S, C, levels=[pion_level], colors=[pion_color], linewidths=0.0)
    br, sig = CS.allsegs[0][0].T
    plt.fill(br, sig, edgecolor='k', facecolor=pion_color, alpha=alpha, linewidth=0.0)

    plt.plot(br, sig, c=pion_color, lw=lw, label=r'Inelastic CRDM ($\pi$)')
    plt.plot([br[1], br[-1]], [sig[1], sig[-1]], c=pion_color, lw=lw)

    CS = plt.contour(B, S, C, levels=[pion_level*LZ_FACTOR], colors=[pion_color], linewidths=0.0)
    br, sig = CS.allsegs[0][0].T
    #plt.fill(br, sig, edgecolor='k', facecolor=pion_color, alpha=alpha, linewidth=0.0)

    plt.plot(br, sig, c=pion_color, lw=lw, ls='--')
    plt.plot([br[1], br[-1]], [sig[1], sig[-1]], c=pion_color, lw=lw, ls='--')

    plt.xlabel(r'$\mathrm{BR}(M \rightarrow X\chi\chi)$')
    plt.ylabel(r'$\sigma_\chi^{\mathrm{\tiny SI}}\,\mathrm{[cm}^2\mathrm{]}$')
    #plt.title(r'Exclusion Limits')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.LogLocator(numticks=6))
    ax.tick_params(which='minor', length=4)
    plt.legend(fontsize=14, loc='lower left')
    axes = plt.axis()
    plt.axis([np.power(10.0, -10.0), 5*np.power(10.0, -3.0), np.power(10.0, -37), np.power(10.0, -27)])
    fontsize = 12
    plt.text(np.power(10.0, -6.35), np.power(10.0, -28.85), 'XENON1T', color=eta_color, fontsize=fontsize, rotation=-43.0)
    plt.text(np.power(10.0, -3.1), np.power(10.0, -34.2), 'LZ', color=eta_color, fontsize=fontsize, rotation=-41.0)
    plt.text(np.power(10.0, -7.45), np.power(10.0, -28.85), 'XENON1T', color=pion_color, fontsize=fontsize, rotation=-43.0)
    plt.text(np.power(10.0, -8.0), np.power(10.0, -30.15), 'LZ', color=pion_color, fontsize=fontsize, rotation=-47.0)
    plt.text(1.1*np.power(10.0, -9.85), np.power(10.0, -30.73), 'XENON1T', color=pospelov_color, fontsize=fontsize, rotation=0.0)
    plt.text(np.power(10.0, -9.5), np.power(10.0, -33.9), r'$m_\chi = 1\,\mathrm{MeV}$')
    plt.text(np.power(10.0, -9.5), np.power(10.0, -34.4), r'$m_{\mathrm{med}} = 10\,\mathrm{MeV}$')
    plt.savefig('plots/limit_no_title.pdf')
