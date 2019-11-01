# File: attenuation.py
#
# implements the different levels of complexity for the mean free path

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

def generate_Earth_df():
    r"""
    Generates Table 2 in `1611.05453 <https://arxiv.org/pdf/1611.05453.pdf>`_ as a pandas dataframe.

    Returns
    -------
    Earth_df : pd.DataFrame
        the dataframe with columns ['Element', 'A', 'mA', 'ncore', 'nmantle', 'Core', 'Mantle']
    """

    Rcore = 3400*np.power(10.0, 5) # cm
    RE = 6378.1*np.power(10.0, 5) # cm
    Vcore = (4/3)*np.pi*np.power(Rcore, 3)
    Vmantle = (4/3)*np.pi*np.power(RE, 3) - Vcore
    MEGeV = 3.3501577*np.power(10.0, 51)
    fcore = 0.32
    fmantle = 0.68

    Element = np.array(['Oxygen', 'Silicon', 'Magnesium', 'Iron', 'Calcium', 'Sodium', 'Sulphur', 'Aluminium'])
    A = np.array([16.0, 28.0, 24.0, 56.0, 40.0, 23.0, 32.0, 27.0])
    mA = np.array([14.9, 26.1, 22.3, 52.1, 37.2, 21.4, 29.8, 25.1])
    nbar = np.array([3.45*np.power(10.0, 22), 1.77*np.power(10.0, 22), 1.17*np.power(10.0, 22), 6.11*np.power(10.0, 22), 7.94*np.power(10.0, 20), 1.47*np.power(10.0, 20), 2.33*np.power(10.0, 21), 1.09*np.power(10.0, 21)])
    Core = np.array([0.0, 0.06, 0.0, 0.855, 0.0, 0.0, 0.019, 0.0])
    Mantle = np.array([0.440, 0.210, 0.228, 0.0626, 0.0253, 0.0027, 0.00025, 0.0235])
    ncore = Core*fcore*MEGeV*np.power(mA, -1.0)*np.power(Vcore, -1.0)
    nmantle = Mantle*fmantle*MEGeV*np.power(mA, -1.0)*np.power(Vmantle, -1.0)
    Earth_df = pd.DataFrame({'Element': Element, 'A': A, 'mA': mA, 'ncore': ncore, 'nmantle': nmantle, 'nbar': nbar, 'Core': Core, 'Mantle': Mantle})
    return Earth_df

def mfp_nocore(sigmaSI, mchi):
    r"""
    Returns the mean free path through the Earth assuming no core/mantle distinction. Data is taken from Table 2 in `1611.05453 <https://arxiv.org/pdf/1611.05453.pdf>`_ and used to find the average number densities in the core. This can be obtained from :py:func:`generate_Earth_df`.

    Parameters
    ----------
    sigmaSI : float
        spin-independent WIMP-nucleon cross-section [:math:`\textrm{cm}^2`]
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]

    Returns
    -------
    mfp_nocore : float
        the mean free path through the Earth assuming no core [:math:`\textrm{km}`]

    Notes
    -----
    To calculate the mean free path we are using the expression,

    .. math:: \ell^{-1}(\sigma_\chi^{\textrm{SI}}) = \sum_{N}{\sigma_{\chi N} \bar{n}_N}

    where :math:`\sigma_{\chi N}` is the coherent scattering cross section which sees an enhancement by a factor :math:`A^2`.

    Examples
    --------
    >>> mfp_nocore(sigmaSI=np.power(10.0, -32), mchi=0.001)
    2211.9344188520954
    """
    mp = 0.938 # GeV
    Earth_df = generate_Earth_df()
    sigma_chi_N = np.power(Earth_df['A'], 2.0)*np.power((Earth_df['mA']*(mchi + mp))*np.power(mp*(mchi + Earth_df['mA']), -1.0), 2.0)
    inv_mfp_core = sigmaSI*np.sum(sigma_chi_N*Earth_df['nbar']*2*Earth_df['mA']*mchi*np.power(mchi + Earth_df['mA'], -2.0))
    return np.power(inv_mfp_core, -1.0)/np.power(10.0, 5)

def mfp_core(sigmaSI, mchi):
    r"""
    Returns the mean free path through the core. Data is taken from Table 2 in `1611.05453 <https://arxiv.org/pdf/1611.05453.pdf>`_ and used to find the average number densities in the core. This can be obtained from :py:func:`generate_Earth_df`.

    Parameters
    ----------
    sigmaSI : float
        spin-independent WIMP-nucleon cross-section [:math:`\textrm{cm}^2`]
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]

    Returns
    -------
    mfp_core : float
        the mean free path through the Earth's core [:math:`\textrm{km}`]

    Notes
    -----
    To calculate the mean free path we are using the expression,

    .. math:: \ell^{-1}(\sigma_\chi^{\textrm{SI}}) = \sum_{N}{\sigma_{\chi N} \bar{n}_N}

    where :math:`\sigma_{\chi N}` is the coherent scattering cross section which sees an enhancement by a factor :math:`A^2`.

    Examples
    --------
    >>> mfp_core(sigmaSI=np.power(10.0, -32), mchi=0.001)
    1421.1672801450084
    """
    mp = 0.938 # GeV
    Earth_df = generate_Earth_df()
    sigma_chi_N = np.power(Earth_df['A'], 2.0)*np.power((Earth_df['mA']*(mchi + mp))*np.power(mp*(mchi + Earth_df['mA']), -1.0), 2.0)
    inv_mfp_core = sigmaSI*np.sum(sigma_chi_N*Earth_df['ncore']*2*Earth_df['mA']*mchi*np.power(mchi + Earth_df['mA'], -2.0))
    return np.power(inv_mfp_core, -1.0)/np.power(10.0, 5)

def mfp_mantle(sigmaSI, mchi):
    r"""
    Returns the mean free path through the mantle. Data is taken from Table 2 in `1611.05453 <https://arxiv.org/pdf/1611.05453.pdf>`_ and used to find the average number densities in the mantle. This can be obtained from :py:func:`generate_Earth_df`.

    Parameters
    ----------
    sigmaSI : float
        spin-independent WIMP-nucleon cross-section [:math:`\textrm{cm}^2`]
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]

    Returns
    -------
    mfp_mantle : float
        the mean free path through the Earth's mantle [:math:`\textrm{km}`]

    Notes
    -----
    To calculate the mean free path we are using the expression,

    .. math:: \ell^{-1}(\sigma_\chi^{\textrm{SI}}) = \sum_{N}{\sigma_{\chi N} \bar{n}_N}

    where :math:`\sigma_{\chi N}` is the coherent scattering cross section which sees an enhancement by a factor :math:`A^2`.

    Examples
    --------
    >>> mfp_mantle(sigmaSI=np.power(10.0, -32), mchi=0.001)
    1421.1672801450084
    """
    mp = 0.938 # GeV
    Earth_df = generate_Earth_df()
    sigma_chi_N = np.power(Earth_df['A'], 2.0)*np.power((Earth_df['mA']*(mchi + mp))*np.power(mp*(mchi + Earth_df['mA']), -1.0), 2.0)
    inv_mfp_core = sigmaSI*np.sum(sigma_chi_N*Earth_df['nmantle']*2*Earth_df['mA']*mchi*np.power(mchi + Earth_df['mA'], -2.0))
    return np.power(inv_mfp_core, -1.0)/np.power(10.0, 5)

def TzMin(mchi=0.001, TN=4.9*np.power(10.0, -6.0), mN=115.3909):
    r"""
    Returns the minimum kinetic energy that can illicit a response in the detector. This will then be used to calculate the maximum cross-section for which the Earth may be considered transparent.

    Parameters
    ----------
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]
    TN : float
        minimum recoil energy in the detector [:math:`\textrm{GeV}`]
    mN : float
        mass of the target nucleus [:math:`\textrm{GeV}`] (see `this page <http://hyperphysics.phy-astr.gsu.edu/hbase/pertab/xe.html>`_ for Xenon data)

    Returns
    -------
    TzMin : float
        minimum kinetic energy to produce response in detector

    Notes
    -----
    We are using the definition in equation (2) of `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`_,

    .. math:: T_z^{\textrm{min}} = \left(\frac{T_N}{2} - m_\chi\right)\left(1 \pm \sqrt{1 + \frac{2 T_N (m_N + m_\chi)^2}{m_N(2m_\chi - T_N)^2}}\right)

    where the :math:`+` sign applies for :math:`T_N > 2 m_\chi` and vice versa.

    Examples
    --------
    >>> TzMin(mchi=0.001, TN=4.3*np.power(10.0, -6.0), mN=115.3909)
    0.014784749267583662
    """
    if TN == 2*mchi:
        return 0.0
    elif TN > 2*mchi:
        return (0.5*TN - mchi)*(1 + np.sqrt(1 + (2*TN*np.power(mchi + mN, 2.0))/(mN*np.power(2*mchi - TN, 2.0))))
    else:
        return (0.5*TN - mchi)*(1 - np.sqrt(1 + (2*TN*np.power(mchi + mN, 2.0))/(mN*np.power(2*mchi - TN, 2.0))))

def TchiDenom(z, l, Tz, mchi):
    r"""
    Returns the denominator for the dark matter kinetic energy in terms of the energy at some depth :math:`z`. This should be used to solve for the maximum line of sight distance through the Earth given a certain cross-section.

    Parameters
    ----------
    z : float
        line of sight distance [:math:`\textrm{km}`]
    l : float
        mean free path [:math:`\textrm{km}`]
    Tz : float
        kinetic energy at depth :math:`z` [:math:`\textrm{GeV}`]
    mchi : float
        mass of dark matter particle [:math:`\textrm{GeV}`]

    Returns
    -------
    TchiDenom : float
        denominator of equation (12) in `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`_.

    Notes
    -----
    The function being implemented is,

    .. math:: 2 m_\chi + T_\chi^z - T_\chi^z e^{z/\ell}

    where :math:`\ell` is the mean free path along the line of sight.
    """
    return 2*mchi + Tz - Tz*np.exp(z*np.power(l, -1.0))

def max_z(sigmaSI, mchi, type='simple'):
    r"""
    Uses :py:func:`scipy.optimize.fsolve` to find the point where the denominator in :py:func:`TchiDenom` vanishes and hence the maximum line of sight depth.

    Parameters
    ----------
    sigmaSI : float
        the spin-independent cross section, :math:`\sigma_\chi^{\textrm{SI}}` [:math:`\textrm{cm}^2`]
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]

    Returns
    -------
    max_z : float
        estimated maximum line of sight distance [:math:`\textrm{km}`]

    Notes
    -----
    We are looking to solve for :math:`z` in the following,

    .. math:: 2 m_\chi + T_\chi^z - T_\chi^z e^{z/\ell} = 0
    """
    Tz = TzMin(mchi)
    if type == 'simple':
        l = mfp_nocore(sigmaSI, mchi)
    max_z = fsolve(TchiDenom, x0=0.0, args=(l, Tz, mchi))
    return max_z

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # sigmaSI = np.power(10.0, np.linspace(-33.0, -28.0, 1000))
    # mchi = 0.001
    # mfp_core = mfp_core(sigmaSI, mchi)
    # mfp_mantle = mfp_mantle(sigmaSI, mchi)
    # mfp_nocore = mfp_nocore(sigmaSI, mchi)
    # plt.figure(figsize=(6,6))
    # plt.plot(sigmaSI, mfp_core, label='Core')
    # plt.plot(sigmaSI, mfp_mantle, label='Mantle')
    # plt.plot(sigmaSI, mfp_nocore, label='Simplified (No Core)')
    # plt.plot([sigmaSI[0], sigmaSI[-1]], [1.4, 1.4], c='k', ls='--')
    # plt.plot([sigmaSI[0], sigmaSI[-1]], [2*6378.1, 2*6378.1], c='k', ls='--')
    # plt.fill_between([sigmaSI[0], sigmaSI[-1]], [1.4, 1.4], [2*6378.1, 2*6378.1], color='k', alpha=0.1)
    # plt.title(r'Mean Free Path ($m_\chi = 0.001 \, \textrm{GeV}$)')
    # plt.xlabel(r'$\sigma_{\chi}^{\textrm{SI}}\,\textrm{[cm}^2\textrm{]}$')
    # plt.ylabel(r'$\ell(\sigma_\chi^{\textrm{SI}}, m_\chi)\,\textrm{[km]}$')
    # plt.text(np.power(10.0, -32.7), np.power(10.0, 4.15), r'$2R_E$', fontsize=12)
    # plt.text(np.power(10.0, -32.7), np.power(10.0, 0.2), r'$h_d = 1.4\,\textrm{km}$', fontsize=12)
    # plt.legend()
    # plt.gca().set_xscale('log')
    # plt.gca().set_yscale('log')
    # plt.savefig('plots/mfp.pdf')

    RE = 6378.1
    zarr = np.linspace(0.0, 2.0, 1000)
    sigmaarr = np.logspace(-30.0, -27.0, 1000)
    S, Z = np.meshgrid(sigmaarr, zarr)
    L = mfp_nocore(S, mchi=0.001)
    D = TchiDenom(Z, L, TzMin(mchi=0.001), mchi=0.001)
    mask = (D > 0)
    plt.figure()
    plt.contourf(S, Z, D*mask)
    plt.xlabel(r'$\sigma_{\chi}^{\textrm{\small SI}}\,\textrm{[cm}^2\textrm{]}$')
    plt.ylabel(r'$z\,\textrm{[km]}$')
    plt.title(r'$2m_\chi + T_\chi^{z, \textrm{\small min}}(1 - e^{z/\ell(\sigma_{\chi}^{\textrm{\tiny SI}})})$')
    plt.gca().set_xscale('log')
    #plt.colorbar()
    plt.savefig('plots/denomcontourzoom.pdf')

    RE = 6378.1
    zarr = np.linspace(0.0, 2*RE, 1000)
    sigmaarr = np.logspace(-34.0, -30.0, 1000)
    S, Z = np.meshgrid(sigmaarr, zarr)
    L = mfp_nocore(S, mchi=0.001)
    D = TchiDenom(Z, L, TzMin(mchi=0.001), mchi=0.001)
    mask = (D > 0)
    plt.figure()
    plt.contourf(S, Z/10**3, D*mask)
    plt.xlabel(r'$\sigma_{\chi}^{\textrm{\small SI}}\,\textrm{[cm}^2\textrm{]}$')
    plt.ylabel(r'$z\,\textrm{[}10^3\,\textrm{km]}$')
    plt.title(r'$2m_\chi + T_\chi^{z, \textrm{\small min}}(1 - e^{z/\ell(\sigma_{\chi}^{\textrm{\tiny SI}})})$')
    plt.gca().set_xscale('log')
    #plt.colorbar()
    plt.savefig('plots/denomcontour.pdf')

    from attenuation import zstar
    sigmalist = np.array([np.power(10.0, -32), 5*np.power(10.0, -32), np.power(10.0, -30), np.power(10.0, -29)])
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    idx = 1
    cmap = 'Blues_r'
    lines = '#D1495B'
    for sigma in sigmalist:
        H, T = np.meshgrid(np.linspace(0.0, 180.0, 1000), np.linspace(0.0, np.pi, 1000))
        Z = zstar(H, T)
        testzarr = np.linspace(0.0, 2*6378.1, 10000)
        mfp = mfp_nocore(sigma, mchi=0.001)
        denomarr = TchiDenom(Tz=TzMin(), l=mfp, z=testzarr, mchi=0.001)
        mask = (denomarr > 0)
        if idx == 1:
            axes[0, 0].contour(H, T, Z, levels=[testzarr[mask].max()], colors=[lines])
            axes[0, 0].contourf(H, T, Z, cmap=cmap)
            axes[0, 0].text(90.0, 2.8, r'$\sigma = 10^{-32}\,\textrm{cm}^2$')
            axes[0, 0].set_ylabel(r'$\theta$')
        if idx == 2:
            axes[0, 1].contour(H, T, Z, levels=[testzarr[mask].max()], colors=[lines])
            axes[0, 1].contourf(H, T, Z, cmap=cmap)
            axes[0, 1].text(70.0, 2.8, r'$\sigma = 5 \times 10^{-32}\,\textrm{cm}^2$')
        if idx == 3:
            axes[1, 0].contour(H, T, Z, levels=[testzarr[mask].max()], colors=[lines])
            axes[1, 0].contourf(H, T, Z, cmap=cmap)
            axes[1, 0].text(90.0, 2.8, r'$\sigma = 10^{-30}\,\textrm{cm}^2$')
            axes[1, 0].set_ylabel(r'$\theta$')
            axes[1, 0].set_xlabel(r'$h\,\textrm{[km]}$')
        if idx == 4:
            axes[1, 1].contour(H, T, Z, levels=[testzarr[mask].max()], colors=[lines])
            axes[1, 1].contourf(H, T, Z, cmap=cmap)
            axes[1, 1].text(90.0, 2.8, r'$\sigma = 10^{-29}\,\textrm{cm}^2$')
            axes[1, 1].set_xlabel(r'$h\,\textrm{[km]}$')
        idx += 1
    plt.savefig('plots/integrationregion.pdf')
