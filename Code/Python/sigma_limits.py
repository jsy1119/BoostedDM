# File: sigma_limits.py

import numpy as np
import scipy.stats
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time
from matplotlib import ticker
import matplotlib

from three_body import chi_energies
from attenuation import Gdet
from mean_free_path import TzMin
from limits import TNBound, TrNMax

def decay_eta(mchi, mv, br=5*np.power(10.0, -2.0)):
    r"""
    Reads in the eta energies from eta_energies.npy and eta_weights.npy and carries out the kinematics for the decay :math:`\eta \rightarrow \pi\chi\chi`. The energies and weights for the DM are then returned as two numpy arrays.

    Parameters
    ----------
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]
    mv : float
        mass of the mediator [:math:`\textrm{GeV}`]
    br : float
        branching ratio :math:`\textrm{BR}(\eta \rightarrow \pi\chi\chi)`

    Returns
    -------
    chienergies, chiweights : np.array, np.array
        arrays of DM energies and weights
    """
    meta = 0.547862 # GeV
    eta_energies = np.load('eta_energies.npy')
    eta_weights = np.load('eta_weights.npy')

    dm_energies = np.array([])
    dm_weights = np.array([])

    for idx in range(0, len(eta_energies)):
        new_energies = chi_energies(eta_energies[idx], meta, mv, mchi) - mchi
        dm_energies = np.append(dm_energies, new_energies)
        dm_weights = np.append(dm_weights, np.array([eta_weights[idx], eta_weights[idx]]))
    return dm_energies, br*dm_weights

def decay_pi(mchi, mv, br=6*np.power(10.0, -4.0)):
    r"""
    Reads in the eta energies from eta_energies.npy and eta_weights.npy and carries out the kinematics for the decay :math:`\eta \rightarrow \pi\chi\chi`. The energies and weights for the DM are then returned as two numpy arrays.

    Parameters
    ----------
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]
    mv : float
        mass of the mediator [:math:`\textrm{GeV}`]
    br : float
        branching ratio :math:`\textrm{BR}(\eta \rightarrow \pi\chi\chi)`

    Returns
    -------
    chienergies, chiweights : np.array, np.array
        arrays of DM energies and weights
    """
    mpi = 0.1349770 # GeV
    pi_energies = np.load('pi_energies.npy')
    pi_weights = np.load('pi_weights.npy')

    dm_energies = np.array([])
    dm_weights = np.array([])

    for idx in range(0, len(pi_energies)):
        new_energies = chi_energies(pi_energies[idx], mpi, mv, mchi) - mchi
        dm_energies = np.append(dm_energies, new_energies)
        dm_weights = np.append(dm_weights, np.array([pi_weights[idx], pi_weights[idx]]))
        print('Completed {} out of {} energies'.format(idx + 1, len(pi_energies)), end='\r')
    return dm_energies, br*dm_weights

def get_dm_kernel(dm_energies, dm_weights):
    r"""
    Obtains the kernel and the normalisation factor for the dark matter flux from pions and eta particle decays. This can then be called directly on an array of energies as `kernel.evaluate(energies)`.

    Parameters
    ----------
    dm_energies, dm_weights : np.array, np.array
        pre-computed energies and weights using :py:func:`decay_eta`

    Returns
    -------
    kernel : scipy.stats.gaussian_kde
        the kernel, which can be called directly on an array of energies
    normalisation factor : float
        the normalisation factor for the flux, this is because the scipy.stats.gaussian_kde is normalised by default
    """
    bw = np.power(10.0, -1.5)
    kernel = scipy.stats.gaussian_kde(dataset=dm_energies, weights=dm_weights, bw_method=bw)
    counts, _ = np.histogram(dm_energies, bins=1, weights=dm_weights)
    norm = counts[0]
    return kernel, norm

def get_eta_rate(mchi, mv, br=5*np.power(10.0, -2.0), dTchi=np.power(10.0, -3.0), TchiMax=np.power(10.0, 0.0)):
    r"""
    Computes the integral in the denominator of equation (16) in `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`_. Computes it in the case of both the pions and the eta fluxes. [version 2]

    Parameters
    ----------
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]
    mv : float
        mediator mass [:math:`\textrm{GeV}`]
    br : float
        branching ratio :math:`\textrm{BR}(\eta \rightarrow \pi \chi\chi)`
    dTchi : float
        DM energy resolution [:math:`\textrm{GeV}`]
    TchiMax : float
        Maximum integration energy

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
    dm_energies, dm_weights = decay_eta(mchi=mchi, mv=mv, br=br)
    kernel, norm = get_dm_kernel(dm_energies, dm_weights)
    TchiMin = TzMin(mchi=mchi)
    TchiArr = np.arange(TchiMin, TchiMax, dTchi)
    TNBoundArr = TNBound(TchiArr, mchi=mchi)
    dTNArr = TNBoundArr - 4.9*np.power(10.0, -6.0)
    fluxArr = norm*kernel.evaluate(TchiArr)
    TrNArr = TrNMax(TchiArr, mchi)
    integrands = fluxArr*np.power(TrNArr, -1.0)*dTNArr*dTchi
    integral = integrands.sum()
    print('\n--------\nResults:\n--------\n\nIntegral: {}, mchi = {}\n'.format(integral, mchi))
    return integral

def run_sigma_mchi_eta(mchiArr, sigmaArr, mv, save_name):
    r"""
    Runs across all energies and cross sections.

    Parameters
    ----------
    mchiArr : np.array
        array of DM masses [:math:`\textrm{GeV}`]
    sigmaArr : np.array
        array of cross sections [:math:`\textrm{cm}^2`]
    mv : float
        mediator mass [:math:`\textrm{GeV}`]
    """
    sigma_size = len(sigmaArr)
    mchi_size = len(mchiArr)
    print('Stage 1. - Computing Rates')
    time.sleep(2)
    ratesArr = np.array(Parallel(n_jobs=-1)(delayed(get_eta_rate)(mchi=mchi, mv=mv) for mchi in mchiArr))
    print('Rates Array:\n')
    print(ratesArr)
    ratesArr = np.tile(ratesArr, sigma_size)
    print('Stage 1. - Computing Rates (COMPLETE)')
    time.sleep(2)
    mchiArr = np.tile(mchiArr, sigma_size)
    sigmaArr = np.repeat(sigmaArr, mchi_size)
    print('Stage 2. - Computing GdetArr')
    time.sleep(2)
    GdetArr = np.array(Parallel(n_jobs=-1)(delayed(Gdet)(sigmachi=sigmachi, mchi=mchi) for (sigmachi, mchi) in zip(sigmaArr, mchiArr)))
    print('Gdet Array:\n')
    print(GdetArr)
    print('Stage 2. - Computing GdetArr (COMPLETE)')
    time.sleep(2)
    print('Stage 3. - Compute product')
    time.sleep(2)
    mXe = 115.3909 # GeV
    mp = 0.938272 # GeV
    productArr = np.power(mchiArr + mp, 2.0)*np.power(mchiArr + mXe, -2.0)*sigmaArr*GdetArr*ratesArr
    productArr = np.reshape(productArr, (-1, mchi_size))
    print('Product Array:\n')
    print(productArr)
    print('Stage 3. - Compute product (COMPLETE)')
    time.sleep(2)
    print('Stage 4. - Save array')
    time.sleep(2)
    np.save('{}.npy'.format(save_name), productArr)
    print('Stage 4. - Save array (COMPLETE)')

def get_pi_rate(mchi, mv, br=6*np.power(10.0, -4.0), dTchi=np.power(10.0, -3.0), TchiMax=np.power(10.0, 0.0)):
    r"""
    Computes the integral in the denominator of equation (16) in `1810.10543 <https://arxiv.org/pdf/1810.10543.pdf>`_. Computes it in the case of both the pions and the eta fluxes. [version 2]

    Parameters
    ----------
    mchi : float
        mass of the dark matter particle [:math:`\textrm{GeV}`]
    mv : float
        mediator mass [:math:`\textrm{GeV}`]
    br : float
        branching ratio :math:`\textrm{BR}(\eta \rightarrow \pi \chi\chi)`
    dTchi : float
        DM energy resolution [:math:`\textrm{GeV}`]
    TchiMax : float
        Maximum integration energy

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
    dm_energies, dm_weights = decay_pi(mchi=mchi, mv=mv, br=br)
    kernel, norm = get_dm_kernel(dm_energies, dm_weights)
    TchiMin = TzMin(mchi=mchi)
    TchiArr = np.arange(TchiMin, TchiMax, dTchi)
    TNBoundArr = TNBound(TchiArr, mchi=mchi)
    dTNArr = TNBoundArr - 4.9*np.power(10.0, -6.0)
    fluxArr = norm*kernel.evaluate(TchiArr)
    TrNArr = TrNMax(TchiArr, mchi)
    integrands = fluxArr*np.power(TrNArr, -1.0)*dTNArr*dTchi
    integral = integrands.sum()
    print('\n--------\nResults:\n--------\n\nIntegral: {}, mchi = {}\n'.format(integral, mchi))
    return integral

def run_sigma_mchi_pi(mchiArr, sigmaArr, mv, save_name):
    r"""
    Runs across all energies and cross sections.

    Parameters
    ----------
    mchiArr : np.array
        array of DM masses [:math:`\textrm{GeV}`]
    sigmaArr : np.array
        array of cross sections [:math:`\textrm{cm}^2`]
    mv : float
        mediator mass [:math:`\textrm{GeV}`]
    """
    sigma_size = len(sigmaArr)
    mchi_size = len(mchiArr)
    print('Stage 1. - Computing Rates')
    time.sleep(2)
    ratesArr = np.array(Parallel(n_jobs=-1)(delayed(get_pi_rate)(mchi=mchi, mv=mv) for mchi in mchiArr))
    print('Rates Array:\n')
    print(ratesArr)
    ratesArr = np.tile(ratesArr, sigma_size)
    print('Stage 1. - Computing Rates (COMPLETE)')
    time.sleep(2)
    mchiArr = np.tile(mchiArr, sigma_size)
    sigmaArr = np.repeat(sigmaArr, mchi_size)
    print('Stage 2. - Computing GdetArr')
    time.sleep(2)
    GdetArr = np.array(Parallel(n_jobs=-1)(delayed(Gdet)(sigmachi=sigmachi, mchi=mchi) for (sigmachi, mchi) in zip(sigmaArr, mchiArr)))
    print('Gdet Array:\n')
    print(GdetArr)
    print('Stage 2. - Computing GdetArr (COMPLETE)')
    time.sleep(2)
    print('Stage 3. - Compute product')
    time.sleep(2)
    mXe = 115.3909 # GeV
    mp = 0.938272 # GeV
    productArr = np.power(mchiArr + mp, 2.0)*np.power(mchiArr + mXe, -2.0)*sigmaArr*GdetArr*ratesArr
    productArr = np.reshape(productArr, (-1, mchi_size))
    print('Product Array:\n')
    print(productArr)
    print('Stage 3. - Compute product (COMPLETE)')
    time.sleep(2)
    print('Stage 4. - Save array')
    time.sleep(2)
    np.save('{}.npy'.format(save_name), productArr)
    print('Stage 4. - Save array (COMPLETE)')

def contour_level(experiment):
    r"""
    Gets the contour level for the different experiments (Xenon1T and LZ).

    Parameters
    ----------
    experiment : str ('xenon' or 'lz')
        choice of experiment

    Returns
    -------
    contour_level : float
        contour level

    Notes
    -----
    Implementing the factor,

    .. math:: \kappa (\bar{v}\rho_{\textrm{DM}})^{\textrm{loc}} \left(\frac{\sigma^{\textrm{SI, lim}}_{\textrm{DM}}}{m_{\textrm{DM}}}\right)^{\textrm{Xe/LZ}}_{m_{\textrm{DM}} \rightarrow \infty}
    """
    k = 0.23
    vbar = 250*np.power(10.0, 5) # cm s^-1
    rhodm = 0.3 # GeV cm^-3
    if experiment == 'xenon':
        sigmalim = 8.3*np.power(10.0, -49) # cm^2 GeV^-1
    elif experiment == 'lz':
        sigmalim = 1.9*np.power(10.0, -50)
    return k*vbar*rhodm*sigmalim

if __name__ == '__main__':
    import pandas as pd
    eta_color = '#D81159'
    pion_color = '#01295F'
    pion_color = '#29AB87'
    mchi_eta_size = 10
    sigmachi_eta_size = 50
    mchi_pi_size = 10
    sigmachi_pi_size = 50
    mv = 0.1
    mchi_eta_Arr = np.logspace(-4.0, np.log10(mv/2.0), mchi_eta_size)
    sigma_eta_Arr = np.logspace(-36.0, -27.0, sigmachi_eta_size)
    mchi_pi_Arr = np.logspace(-4.0, np.log10(mv/2.0), mchi_pi_size)
    sigma_pi_Arr = np.logspace(-36.0, -27.0, sigmachi_pi_size)
    #run_sigma_mchi_pi(mchi_pi_Arr, sigma_pi_Arr, mv, 'new_product_pi')
    #run_sigma_mchi_eta(mchi_eta_Arr, sigma_eta_Arr, mv, 'new_product_eta')
    product_eta_Arr = np.load('new_product_eta.npy')
    product_pi_Arr = np.load('new_product_pi.npy')
    xenon_level = contour_level('xenon')
    lz_level = contour_level('lz')
    print('xenon level = {}'.format(xenon_level))
    print('lz level = {}'.format(lz_level))
    M_eta, S_eta = np.meshgrid(mchi_eta_Arr, sigma_eta_Arr)
    M_pi, S_pi = np.meshgrid(mchi_pi_Arr, sigma_pi_Arr)
    plt.figure(figsize=(6,6))
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

    constraint_alpha = 0.1
    constraint_color = '#62656E'
    pospelov_color = '#FF8514'
    fs = 10

    # CMB
    cmb_df = pd.read_csv('constraints/CMB.csv', header=None, names=['mchi', 'sig'])
    plt.plot(cmb_df['mchi'], cmb_df['sig'], c=constraint_color, lw=1.0, ls='-')
    plt.fill_between(cmb_df['mchi'], cmb_df['sig'], np.power(10.0, -24.0), color=constraint_color, alpha=constraint_alpha)
    plt.text(np.power(10.0, -2.9), 1.3*cmb_df['sig'][0], 'CMB', fontsize=fs, color=constraint_color)

    # CRESST
    cresst_df = pd.read_csv('constraints/CRESST.csv', header=None, names=['mchi', 'sig'])
    plt.plot(cresst_df['mchi'], cresst_df['sig'], c=constraint_color, lw=1.0, ls='-')
    plt.fill(cresst_df['mchi'], cresst_df['sig'], color=constraint_color, alpha=constraint_alpha)
    plt.text(0.7*0.8373491973160315, 1.5*2.546132202621052e-34, 'CRESST 17', fontsize=fs, color=constraint_color, rotation=-12.5)
    # Gas Cloud Cooling
    gas_df = pd.read_csv('constraints/GasCloudCooling.csv', header=None, names=['mchi', 'sig'])
    plt.plot(gas_df['mchi'], gas_df['sig'], c=constraint_color, lw=1.0, ls='-')
    plt.fill_between(gas_df['mchi'], gas_df['sig'], np.power(10.0, -24.0), color=constraint_color, alpha=constraint_alpha)
    plt.text(np.power(10.0, -2.9), 1.05*gas_df['sig'][0], 'GAS CLOUD COOLING', fontsize=fs, color=constraint_color, rotation=-19.0)

    # MW Satellites
    mw_df = pd.read_csv('constraints/MWSatellites.csv', header=None, names=['mchi', 'sig'])
    plt.plot(mw_df['mchi'], mw_df['sig'], c=constraint_color, lw=1.0, ls='-')
    plt.fill_between(mw_df['mchi'], mw_df['sig'], np.power(10.0, -24.0), color=constraint_color, alpha=constraint_alpha)
    plt.text(np.power(10.0, -2.9), np.power(10.0, -28.25), 'MW SATELLITES', fontsize=fs, color=constraint_color, rotation=5.0)

    # POSPELOV
    pos_df = pd.read_csv('constraints/pospelov.csv', header=None, names=['mchi', 'sig'])
    plt.plot(pos_df['mchi'], pos_df['sig'], c=pospelov_color, lw=1.0, ls='-')
    plt.fill(pos_df['mchi'], pos_df['sig'], edgecolor='k', facecolor=pospelov_color, alpha=constraint_alpha, linewidth=0.0)
    plt.text(np.power(10.0, -2.9), 1.5*1.79199e-31, 'ECRDM (XENON1T)', fontsize=fs, color=pospelov_color, rotation=0.0)

    # ETA LZ
    CS = plt.contour(M_eta, S_eta, product_eta_Arr, levels=[lz_level], colors=[eta_color], linestyles='-', linewidths=0.0)
    mclower, siglower = CS.allsegs[0][0].T
    mcupper, sigupper = CS.allsegs[0][1].T
    plt.plot(mclower, siglower, c=eta_color, lw=1.0, linestyle='--')
    plt.plot(mcupper, sigupper, c=eta_color, lw=1.0, linestyle='--')
    plt.plot([mclower[-1], mcupper[0]], [siglower[-1], sigupper[0]], c=eta_color, lw=1.0, linestyle='--')
    plt.text(np.power(10.0, -2.9), 1.3*siglower[0], 'ICRDM ($\eta$) (LZ)', fontsize=fs, color=eta_color, rotation=0.0)

    # ETA XENON1T
    CS = plt.contour(M_eta, S_eta, product_eta_Arr, levels=[xenon_level], colors=[eta_color], linestyles='-', linewidths=0.0)
    mclower, siglower = CS.allsegs[0][0].T
    mcupper, sigupper = CS.allsegs[0][1].T
    plt.fill(np.append(mclower, mcupper), np.append(siglower, sigupper), edgecolor='k', facecolor=eta_color, alpha=0.3, linewidth=0.0)
    plt.plot(mclower, siglower, c=eta_color, lw=1.0)
    plt.plot(mcupper, sigupper, c=eta_color, lw=1.0)
    plt.plot([mclower[-1], mcupper[0]], [siglower[-1], sigupper[0]], c=eta_color, lw=1.0)
    plt.text(np.power(10.0, -2.9), 1.3*siglower[0], 'ICRDM ($\eta$) (XENON1T)', fontsize=fs, color=eta_color)

    # PI LZ
    CS = plt.contour(M_pi, S_pi, product_pi_Arr, levels=[lz_level], colors=[pion_color], linestyles='-', linewidths=0.0)
    mclower, siglower = CS.allsegs[0][0].T
    mcupper, sigupper = CS.allsegs[0][1].T
    plt.plot(mclower, siglower, c=pion_color, lw=1.0, linestyle='--')
    plt.plot(mcupper, sigupper, c=pion_color, lw=1.0, linestyle='--')
    plt.plot([mclower[-1], mcupper[0]], [siglower[-1], sigupper[0]], c=pion_color, lw=1.0, linestyle='--')
    plt.text(np.power(10.0, -2.9), 0.5*siglower[0], 'ICRDM ($\pi$) (LZ)', fontsize=fs, color=pion_color, rotation=0.0)

    # PI XENON1T
    CS = plt.contour(M_pi, S_pi, product_pi_Arr, levels=[xenon_level], colors=[pion_color], linestyles='-', linewidths=0.0)
    mclower, siglower = CS.allsegs[0][0].T
    mcupper, sigupper = CS.allsegs[0][1].T
    plt.fill(np.append(mclower, mcupper), np.append(siglower, sigupper), edgecolor='k', facecolor=pion_color, alpha=0.3, linewidth=0.0)
    plt.plot(mclower, siglower, c=pion_color, lw=1.0, linestyle='-')
    plt.plot(mcupper, sigupper, c=pion_color, lw=1.0, ls='-')
    plt.plot([mclower[-1], mcupper[0]], [siglower[-1], sigupper[0]], c=pion_color, lw=1.0, ls='-')
    plt.text(np.power(10.0, -2.9), 0.5*siglower[0], r'ICRDM ($\pi$) (XENON1T)', fontsize=fs, color=pion_color)

    # DETAILS
    plt.text(np.power(10.0, -1.0), np.power(10.0, -35.0 + 0.5), r'$m_{\mathrm{med}} = 0.1\,\mathrm{GeV}$', fontsize=14)
    plt.text(np.power(10.0, -1.0), np.power(10.0, -35.5 + 0.5), r'$\mathrm{BR}(\eta \rightarrow \pi \chi \chi) = 5 \times 10^{-3}$', fontsize=14)
    plt.text(np.power(10.0, -1.0), np.power(10.0, -36.2 + 0.5), r'$\mathrm{BR}(\pi \rightarrow \gamma \chi \chi) = 6 \times 10^{-4}$', fontsize=14)

    plt.xlabel(r'$m_\chi\,\mathrm{[GeV]}$')
    plt.ylabel(r'$\sigma_\chi^{\mathrm{SI}}\,\mathrm{[cm}^2\mathrm{]}$')
    #plt.title(r'Exclusion Limits')
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    axes = plt.axis()
    plt.axis([np.power(10.0, -3.0), np.power(10.0, 1.0), axes[2], np.power(10.0, -24.0)])
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=200)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(which='minor', length=4)
    plt.savefig('plots/sigma_mchi_eta_pi_no_title.pdf')
