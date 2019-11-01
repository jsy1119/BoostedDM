# File: atmos_dm_flux.py
#
# Computing the atmospheric dark matter flux

import numpy as np
import scipy.stats
from joblib import Parallel, delayed

from crmc_utils import sigmapN, atmos_generate_all_LHE, remove_all_LHE, save_all_energies, load_all_energies
from proton_flux import dPhiPatmosdT

from air_density import suppression_factor, rho

def kdemultifit(energies, weights, cuts, bandwidths, branching_ratio, particle):
    r"""
    Returns the normalised multi fit for the kde given an array of energies and weights. This allows for the user to choose multiple energy cuts and apply different bandwidth fits in each bin.

    Parameters
    ----------
    energies : np.array
        array of energies to fit
    weights : np.array
        array of weights
    cuts : np.array[L + 1]
        energy intervals to fit the kde on
    bandwidths : np.array[L]
        array of bandwidths to use in each interval
    branching_ratio : float
        branching ratio for :math:`\pi^0 \rightarrow \gamma \chi \chi`
    particle : str
        if particle is 'dm' pre multiplies by branching_ratio, else doesn't

    Returns
    -------
    kernel_evals, kernel_pdf : np.array, np.array
        evaluation points for the pdf [default = 100 points] and a normalised array that has been pre-multiplied by the branching_ratio and energies

    Examples
    --------
    >>> energies = np.random.normal(loc=0.0, scale=1.0, size=1000)
    >>> weights = np.random.uniform(0.0, 1.0, 1000)
    >>> cuts = np.array([-5.0, -1.5, 1.5, 5.0])
    >>> bandwidths = np.array([1.0, 0.5, 3.0])
    >>> kde_evals, kde_pdf = kdemultifit(energies, weights, cuts, bandwidths, branching_ratio, particle)
    >>> plt.plot(kde_evals, kde_pdf)
    """
    kde_evals = np.array([])
    kde_pdf = np.array([])
    for idx in range(0, len(bandwidths)):
        temp_evals, temp_pdf = kdekernel(energies, weights, cuts[idx], cuts[idx + 1], branching_ratio, particle, bandwidths[idx])
        kde_evals = np.append(kde_evals, temp_evals[:-1])
        kde_pdf = np.append(kde_pdf, temp_pdf[:-1])
    return kde_evals, kde_pdf

def kdekernel(energies, weights, Emin, Emax, branching_ratio, particle, bw):
    r"""
    Returns the normalised kde kernel that fits a set of energies and weights for a given bandwidth. Used to provide multiple fits across the energy range of interest since variable bandwidth does not appear to be an option.

    Parameters
    ----------
    energies : np.array
        array of energies to fit
    weights : np.array
        array of weights
    Emin : float
        minumum energy to fit the kde on
    Emax : float
        maximum energy to fit the kde on
    branching_ratio : float
        branching ratio for :math:`\pi^0 \rightarrow \gamma \chi \chi`
    particle : str
        if particle is 'dm' pre multiplies by branching_ratio, else doesn't
    bw : float or None
        chosen bandwidth, if None, uses the default `scipy.stats.gaussian_kde` choice

    Returns
    -------
    kernel_evals, kernel_pdf : np.array, np.array
        evaluation points for the pdf [default = 100 points] and a normalised array that has been pre-multiplied by the branching_ratio and energies

    Examples
    --------
    >>> energies = np.random.normal(loc=0.0, scale=1.0, size=1000)
    >>> weights = np.random.uniform(0.0, 1.0, 1000)
    >>> kde_evals, kde_pdf = kdekernel(data, weights, -3.0, 3.0, 10**(-6), 'dm', bw=0.1)
    >>> plt.plot(kde_evals, kde_pdf)
    """
    counts, kdebins = np.histogram(energies, bins=1, weights=weights)
    # Normalise the kernel estiamte as it generates a density
    area = counts[0]

    kernel = scipy.stats.gaussian_kde(dataset=energies, weights=weights, bw_method=bw)

    log_kernel_evals = np.linspace(np.log10(Emin), np.log10(Emax), 100)
    kernel_evals = np.power(10.0, log_kernel_evals)
    if particle == 'dm':
        kernel_pdf = area*branching_ratio*kernel_evals*kernel.evaluate(kernel_evals)
    elif particle == 'pion':
        kernel_pdf = area*kernel_evals*kernel.evaluate(kernel_evals)
    return kernel_evals, kernel_pdf


def geometrical_factor(h1, h2, npts=1000):
    r"""
    Computes the geometrical factor, :math:`F_g`, obtained by integrating around the whole Earth between a minimum height h1 [:math:`\textrm{km}`] and a maximum height h2 [:math:`\textrm{km}`].

    Parameters
    ----------
    h1 : float
        minimum height [:math:`\textrm{km}`]
    h2 : float
        maximum height [:math:`\textrm{km}`]
    npts : int
        number of integration points

    Returns
    -------
    Fg : float
        geometrical factor [:math:`\textrm{cm}^{-2}`]

    Examples
    --------
    >>> geometrical_factor(h1=5, h2=180, npts=100)
    2.556488314653994e+25
    """
    harr = np.linspace(h1, h2, npts)
    dh = (harr[1] - harr[0])*np.power(10.0, 5.0) # so dh is in [cm]
    Fg = 0.0
    Fg -= 0.5*(geometrical_integrand(harr[0]) + geometrical_integrand(harr[-1]))
    for height in harr:
        Fg += geometrical_integrand(height)
    Fg = Fg*dh
    return Fg

def geometrical_integrand(h):
    r"""
    Returns the value of the integrand for the height integral used to calculate the geometrical factor.

    Parameters
    ----------
    h : float
        height [:math:`\textrm{km}`]

    Returns
    -------
    integrand : float
        integrand [:math:`\textrm{cm}^{-3}`]

    Examples
    --------
    >>> geometrical_integrand(h=10)
    6.437549297644509e+18
    """
    Re = 6378.1 # radius of the Earth [km]
    return ((Re + h)/(2*Re))*rho(h)*suppression_factor(h)*np.log((4*Re*(Re + h) + np.power(h, 2))/(np.power(h, 2)))

def atmos_run_all_energies(evals, nMC, save_directory='/mnt/james/muon/', lhe_directory='/mnt/james/lhe/', crmcinstall='/home/k1893416/crmcinstall/'):
    r"""
    Runs the crmc process across all energies in eval, with nMC monte carlo events at each energy. Generates the .lhe files at each energy which are used to construct the .npy files used in :py:func:`atmos_generate_energies_and_weights`. The .lhe files are then deleted to save memory.

    Parameters
    ----------
    evals : np.array[floats]
        proton energies [:math:`\textrm{GeV}`]
    nMC : int
        total number of monte carlo simulations at each energy
    save_directory : str
        save location for .npy files
    lhe_directory : str
        save location for .lhe files
    crmcinstall : str
        file location for crmc install
    """
    count = 1
    Ntot = len(evals)
    # for Ep in evals:
    #     atmos_generate_all_LHE(Ep, nMC, lhe_directory, crmcinstall)
    #     save_all_energies(Ep, nMC, save_directory=save_directory, pid=14.0, lhe_directory=lhe_directory)
    #     remove_all_LHE(Ep, lhe_directory)
    #
    #     print('\n\nCompleted simulations for {} out of {} energies\n\n'.format(count, Ntot))
    #     count += 1
    Parallel(n_jobs=-1)(delayed(parallel_fn)(Ep, nMC, lhe_directory, crmcinstall, save_directory) for Ep in evals)

def parallel_fn(Ep, nMC, lhe_directory, crmcinstall, save_directory):
    atmos_generate_all_LHE(Ep, nMC, lhe_directory, crmcinstall)
    save_all_energies(Ep, nMC, save_directory=save_directory, pid=14.0, lhe_directory=lhe_directory)
    remove_all_LHE(Ep, lhe_directory)
    print('Completed simulations for Ep = {:.1f} GeV\n'.format(Ep))

def atmos_generate_energies_and_weights(proton_evals, nMC, Fg=None, save_directory='/mnt/james/muon/'):
    r"""
    After running crmc to generate the .npy files in :py:func:`atmos_run_all_energies`, we need to combine them into a large array. When eventually plotted, also need to rescale the heights of the bins by the weighting factor. This can be done by generating a set of weights that is the same length as the array of pion energies. This function returns both the concatenated energies from all Ep runs, as well as the weighting factors. After dividing the heights by the bin widths, these can then be plotted directly onto a histogram to obtain the pion flux.

    Parameters
    ----------
    proton_evals : np.array
        array of evaluation points for proton energies [:math:`\textrm{GeV}`], derive :math:`\Delta T_p` from this
    nMC : int
        total number of Monte Carlo runs at each proton energy
    Fg : float
        geometrical factor calculated in :py:func:`geometrical_factor`
    save_directory : str
        save directory for the .npy files

    Returns
    -------
    energies : np.array
        large array of all energies from all the runs
    weights : np.array
        large array of weights to account for the flux normalisation.

    Notes
    -----
    The exact formula for the weight depends on the proton energy and is given by

    .. math:: W(T_p) = \frac{1}{n_{\textrm{MC}}} \frac{\textrm{d}\phi_p}{\textrm{d}T_p}\Delta T_p

    All pions produced from protons at this energy are given the same weight. Also note that the two arrays are of the same length.
    """
    energies = np.array([])
    weights = np.array([])
    dEp = proton_evals[1] - proton_evals[0]
    if Fg == None:
        Fg = geometrical_factor(0.5, 180)

    for Ep in proton_evals:
        energies_from_Ep = load_all_energies(Ep, save_directory)
        length_weights_from_Ep = len(energies_from_Ep)
        weight = (1/nMC)*sigmapN(Ep)*dPhiPatmosdT(Ep)*dEp
        weights_from_Ep = np.full(length_weights_from_Ep, weight)

        energies = np.append(energies, energies_from_Ep)
        weights = np.append(weights, weights_from_Ep)
    return energies, weights
