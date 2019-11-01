# File: pion_flux.py
#
# Setting up the weightings and bins

import numpy as np
import subprocess

from crmc_utils import sigmapp, generate_all_LHE, remove_all_LHE, save_all_energies, load_all_energies
from proton_flux import dPhiPdT

def proton_evals(dEp, EpMin, EpMax):
    r"""
    Generate array for evaluation points centred on the mid points of the bins starting at :math:`E_p^{\textrm{Min}}` and ending at :math:`E_p^{\textrm{Max}}`.

    Parameters
    ----------
    dEp : float
        step length for proton flux [:math:`\textrm{GeV}`]
    EpMin : float
        minimum proton energy [:math:`\textrm{GeV}`]
    EpMax : float
        maximum proton energy [:math:`\textrm{GeV}`]

    Returns
    -------
    evals : np.array
        evaluation points

    Examples
    --------
    >>> proton_evals(dEp=5.0, EpMin=1.0, EpMax=50.0)
    array([ 3.5,  8.5, 13.5, 18.5, 23.5, 28.5, 33.5, 38.5, 43.5, 48.5])
    """
    energy = EpMin + (dEp/2.0)
    evals = np.array([])
    while energy < EpMax:
        evals = np.append(evals, energy)
        energy += dEp
    return evals

def run_all_energies(evals, nMC, save_directory='/mnt/james/pion/', lhe_directory='/mnt/james/lhe/', crmcinstall='/home/k1893416/crmcinstall/'):
    r"""
    Runs the crmc process across all energies in eval, with nMC monte carlo events at each energy. Generates the .lhe files at each energy which are used to construct the .npy files. The .lhe files are then deleted to save memory.

    Parameters
    ----------
    evals : np.array[float]
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
    for Ep in evals:
        generate_all_LHE(Ep, nMC, lhe_directory, crmcinstall)
        save_all_energies(Ep, nMC, save_directory=save_directory, pid=111.0, lhe_directory=lhe_directory)
        remove_all_LHE(Ep, lhe_directory)

        print('\n\nCompleted simulations for {} out of {} energies\n\n'.format(count, Ntot))
        count += 1

def generate_energies_and_weights(proton_evals, nMC, nb=1.0, Deff=1.0, save_directory='/mnt/james/pion/'):
    r"""
    After running crmc to generate the .npy files, we need to combine them into a large array. When eventually plotted, also need to rescale the heights of the bins by the weighting factor. This can be done by generating a set of weights that is the same length as the array of pion energies. This function returns both the concatenated energies from all Ep runs, as well as the weighting factors. These can then be plotted directly onto a histogram to obtain the pion flux.

    Parameters
    ----------
    proton_evals : np.array
        array of evaluation points for proton energies [:math:`\textrm{GeV}`], derive dTp from this
    nMC : int
        total number of Monte Carlo runs at each proton energy
    nb : float
        baryon number density [:math:`\textrm{cm}^{-3}`]
    Deff : float
        volume integration factor [:math:`\textrm{kpc}`]
    save_directory : str
        save directory for the .npy files

    Returns:
    energies : np.array
        large array of all energies from all the runs [:math:`\textrm{GeV}`]
    weights : np.array
        large array of weights to account for the flux normalisation [:math:`\textrm{cm}^{-2}\textrm{s}^{-1}\textrm{GeV}^{-1}`]

    Notes
    -----
    The exact formula for the weighting is given by the following,

    .. math:: W(E_p) = \frac{1}{n_{\textrm{MC}}} n_b \left(\frac{D_{\textrm{eff}}}{\textrm{cm}}\right) \sigma^{\textrm{inel.}}_{pp}(E_p)\frac{\textrm{d}\phi_p(E_p)}{\textrm{d}E_p}

    In particualar, note that it only depends on the proton energy, so all particles produced from collisions at the same initial proton energy have the same weight.
    """
    energies = np.array([])
    weights = np.array([])
    dEp = proton_evals[1] - proton_evals[0]

    for Ep in proton_evals:
        energies_from_Ep = load_all_energies(Ep, save_directory)
        length_weights_from_Ep = len(energies_from_Ep)
        weight = (1/nMC)*nb*(Deff*3.086*np.power(10.0, 21))*sigmapp(Ep)*dPhiPdT(Ep)*dEp
        weights_from_Ep = np.full(length_weights_from_Ep, weight)

        energies = np.append(energies, energies_from_Ep)
        weights = np.append(weights, weights_from_Ep)
    return energies, weights




if __name__ == '__main__':
    dEp = 1.0
    EpMin = 100.0
    EpMax = 110.0
    nMC = 100
    proton_evals = proton_evals(dEp, EpMin, EpMax)
    # run_all_energies(proton_evals, nMC)
