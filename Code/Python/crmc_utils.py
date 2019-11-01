# File: crmc_utils.py
#
# contains useful functions for working with output from crmc
# including obtaining the cross section, and extracting info
# from the .lhe event files

import re
import numpy as np
import subprocess
import pylhe

def sigmapp(Ep, type='inelastic', crmcinstall='/home/k1893416/crmcinstall/'):
    r"""
    For a given proton energy, returns the desired cross section in [:math:`\textrm{cm}^2`] (total, elastic, inelastic [default]) using a regular expression.

    Parameters
    ----------
    Ep : float
        incident proton energy [:math:`\textrm{GeV}`]
    type : str
        either 'total', 'elastic', or 'inelastic'
    crmcinstall : str
        location of the crmc install in filesystem

    Returns
    -------
    sigma : float
        cross section [:math:`\textrm{cm}^2`]

    Notes
    -----
    This requires a crmc installation. The output of `crmc` is in :math:`\textrm{mb}`. This is converted into :math:`\textrm{cm}^2` using :math:`1\,\textrm{cm}^2 = 10^{25}\,\textrm{mb}`.

    Examples
    --------
    >>> sigmapp(Ep=14000, type='inelastic')
    4.079e-26
    """
    process = subprocess.Popen(['cd {}; crmc -x -o lhe -n1 -m0 -p{} -P0 -f out.lhe; rm out.lhe'.format(crmcinstall, Ep)], shell=True, stdout=subprocess.PIPE)
    stdout = process.communicate()[0]
    sigmas = re.findall(r'[-+]?\d*\.\d+|\d+', stdout.decode('utf-8'))

    if type == 'inelastic':
        return np.power(10.0, -27)*float(sigmas[-1])
    elif type == 'elastic':
        return np.power(10.0, -27)*float(sigmas[-2])
    elif type == 'total':
        return np.power(10.0, -27)*float(sigmas[-3])
    else:
        print('ERROR: Incorrect type selected, choose either total, elastic, or inelastic.')

def sigmapN(Ep, type='inelastic', crmcinstall='/home/k1893416/crmcinstall/'):
    r"""
    For a given proton energy, returns the desired pN cross section in [:math:`\textrm{cm}^2`] (total, elastic, inelastic [default]) using a regular expression.

    Parameters
    ----------
    Ep : float
        incident proton energy [:math:`\textrm{GeV}`]
    type : str
        either 'total', 'elastic', or 'inelastic'
    crmcinstall : str
        location of the crmc install on system

    Returns
    -------
    sigma : float
        cross section [:math:`\textrm{cm}^2`]

    Notes
    -----
    It is important to note that `crmc` initially returns the cross section in :math:`\textrm{mb}`.

    Examples
    --------
    >>> sigmapN(Ep=14000, type='inelastic')
    3.0908e-25
    """
    process = subprocess.Popen(['cd {}; crmc -x -o lhe -n1 -m0 -p{} -P0 -I70140 -f out.lhe; rm out.lhe'.format(crmcinstall, Ep)], shell=True, stdout=subprocess.PIPE)
    stdout = process.communicate()[0]
    sigmas = re.findall(r'[-+]?\d*\.\d+|\d+', stdout.decode('utf-8'))

    if type == 'inelastic':
        return np.power(10.0, -27)*float(sigmas[-3])
    elif type == 'elastic':
        return np.power(10.0, -27)*float(sigmas[-2])
    elif type == 'total':
        return np.power(10.0, -27)*float(sigmas[-1])
    else:
        print('ERROR: Incorrect type selected, choose either total, elastic, or inelastic.')

def generate_LHE(Ep, seed, directory='/mnt/james/lhe/', crmcinstall='/home/k1893416/crmcinstall/'):
    r"""
    For a given proton energy and seed (for reproducibility), generates an LHE event file with filename [Ep]-[seed].lhe, saved in chosen directory.

    Parameters
    ----------
    Ep : float
        proton energy [:math:`\textrm{GeV}`]
    seed : int
        seed (in range 1 to 1e9)
    directory : str
        save location for lhe file (potentially requires large amount of memory)
    crmcinstall : str
        location of the crmc install
    """
    process = subprocess.Popen(['cd {}; crmc -o lhe -n1 -m0 -s{} -p{} -P0 -f {}{:.2f}-{}.lhe;'.format(crmcinstall, seed, Ep, directory, Ep, seed)], shell=True, stdout=subprocess.PIPE)
    stdout = process.communicate()[0]

def generate_all_LHE(Ep, nMC, directory='/mnt/james/lhe/', crmcinstall='/home/k1893416/crmcinstall/'):
    r"""
    For a given proton energy and number of Monte Carlo events, nMC, generates an LHE event file for each seed in :math:`[1, n_{\textrm{MC}}]` with filename [Ep]-[seed].lhe, saved in chosen directory.

    Parameters
    ----------
    Ep : float
        proton energy [:math:`\textrm{GeV}`]
    nMC : int
        number of Monte Carlo events
    directory : str
        save location for lhe files (requires large amount of memory)
    crmcinstall : str
        location of the crmc install

    Notes
    -----
    For reproducibility purposes, the choice has been made to interate over seeds in the range :math:`[1, n_{\textrm{MC}}]`. If the user would like to randomly generate this seed instead, they should amend the code to sample the seed each time with e.g. `seed = np.random.randint(1, 1e9)`.
    """
    seed = 1
    while seed <= nMC:
        generate_LHE(Ep, seed, directory, crmcinstall)
        print('Completed {} out of {} monte carlo simulations at proton energy {:.2f} GeV'.format(seed, nMC, Ep), end='\r')
        seed += 1
    process = subprocess.Popen(['clear'], shell=True)

def atmos_generate_LHE(Ep, seed, directory='/mnt/james/lhe/', crmcinstall='/home/k1893416/crmcinstall/'):
    r"""
    For a given proton energy and seed (for reproducibility), generates an LHE event file with filename [Ep]-[seed].lhe, saved in chosen directory. Simulates pN collisions instead of pp collisions.

    Parameters
    ----------
    Ep : float
        proton energy [:math:`\textrm{GeV}`]
    seed : int
        seed (in range 1 to 1e9)
    directory : str
        save location for lhe file (potentially requires large amount of memory)
    crmcinstall : str
        location of the crmc install
    """
    process = subprocess.Popen(['cd {}; crmc -o lhe -n1 -m0 -s{} -p{} -P0 -I70140 -f {}{:.2f}-{}.lhe;'.format(crmcinstall, seed, Ep, directory, Ep, seed)], shell=True, stdout=subprocess.PIPE)
    stdout = process.communicate()[0]
    #print(stdout.decode('utf-8'))

def atmos_generate_all_LHE(Ep, nMC, directory='/mnt/james/lhe/', crmcinstall='/home/k1893416/crmcinstall/'):
    r"""
    For a given proton energy and number of Monte Carlo events, nMC, generates an LHE event file for each seed in [1, nMC] with filename [Ep]-[seed].lhe, saved in chosen directory. Simulating pN collisions.

    Parameters
    ----------
    Ep : float
        proton energy [:math:`\textrm{GeV}`]
    nMC : int
        number of Monte Carlo events
    directory : str
        save location for lhe files (potentially requires large amount of memory)
    crmcinstall : str
        location of the crmc install

    Notes
    -----
    In terms of the random sampling, see :py:func:`generate_all_LHE`.
    """
    seed = 1
    while seed <= nMC:
        atmos_generate_LHE(Ep, seed, directory, crmcinstall)
        print('Completed {} out of {} monte carlo simulations at proton energy {:.2f} GeV'.format(seed, nMC, Ep), end='\r')
        seed += 1
    process = subprocess.Popen(['clear'], shell=True)

def remove_all_LHE(Ep, directory='/mnt/james/lhe/'):
    r"""
    The .lhe files take up a lot of memory. It is therefore more efficient to run for one energy, extract the pion energies, and then delete the .lhe files at that energy. They can always be regenerated later using the seed. Deletes all files [Ep]-[seed].lhe in the stated directory.

    Parameters
    ----------
    Ep : float
        proton energy [:math:`\textrm{GeV}`]
    directory : str
        save location for the lhe files
    """
    process = subprocess.Popen(['cd {}; rm {:.2f}*.lhe;'.format(directory, Ep)], shell=True)

def get_particle_energies(Ep, seed, pid=111.0, lhe_directory='/mnt/james/lhe/'):
    r"""
    For a sigle .lhe file, generates an array of the neutral pion energies.

    Parameters
    ----------
    Ep : float
        proton energy [GeV]
    seed : int
        seed used to generate run
    pid : float
        particle id (111.0 for pion, 221.0 for eta)
    lhe_directory : str
        location of lhe files

    Returns
    -------
    energies : np.array
        array of pion/eta energies [:math:`\textrm{GeV}`]
    """
    filename = lhe_directory + '{:.2f}-{}.lhe'.format(Ep, seed)
    events = []

    for event in pylhe.readLHE(filename):
        events.append(event)
    print('--- Parsed {} ---'.format(filename))

    energies = np.array([])
    for particle in events[0]['particles']:
        if particle['id'] == pid:
            energies = np.append(energies, particle['e'])
    return energies

def get_all_energies(Ep, nMC, pid=111.0, lhe_directory='/mnt/james/lhe/'):
    r"""
    After running :math:`n_{\textrm{MC}}` Monte Carlo events at an energy :math:`E_p`, generates an array of all particle energies that can then be binned and rescaled.

    Parameters
    ----------
    Ep : float
        proton energy [:math:`\textrm{GeV}`]
    nMC : int
        total number of Monte Carlo events
    pid : float
        particle id (111.0 for pion, 221.0 for eta)
    lhe_directory : str
        location of lhe files

    Returns
    -------
    energies : np.array
        array of all pion energies for a given energy

    Notes
    -----
    The list of particle energies that is generated refers to *all* particles produced in the Monte Carlo runs at a *fixed* proton energy :math:`E_p` i.e. it ensures that the user knows that all particles in this list arose from processes where the proton had a given energy. These can then be reweighted according to the proton flux at that energy as well as the total cross-section.
    """
    seed = 1
    energies = np.array([])
    while seed <= nMC:
        energies = np.append(energies, get_particle_energies(Ep, seed, pid, lhe_directory))
        seed += 1
    return energies

def save_all_energies(Ep, nMC, save_directory='/mnt/james/pion/', pid=111.0, lhe_directory='/mnt/james/lhe/'):
    r"""
    Saves the array of particle energies from the nMC simulations at proton energy, Ep, to a .npy file titled [Ep].npy located in the save_directory.

    Parameters
    ----------
    Ep : float
        proton energy [:math:`\textrm{GeV}`]
    nMC : int
        total number of Monte Carlo events
    save_directory : str
        save location for .npy files
    pid : float
        particle id (111.0 for pion, 221.0 for eta)
    lhe_directory : str
        location of lhe files

    Notes
    -----
    In terms of the precise origin of these particles, see the explanation in :py:func:`get_all_energies`.
    """
    seed = 1
    energies = np.array([])
    while seed <= nMC:
        energies = np.append(energies, get_particle_energies(Ep, seed, pid, lhe_directory))
        seed += 1
    np.save('{}{:.2f}.npy'.format(save_directory, Ep), energies)
    print('--- Saved array to {}{:.2f}.npy ---'.format(save_directory, Ep))

def load_all_energies(Ep, save_directory):
    r"""
    Returns the array of particle energies from the .npy file titled [Ep].npy located in the save_directory.

    Parameters
    ----------
    Ep : float
        proton energy [:math:`\textrm{GeV}`]
    save_directory : str
        save location for .npy files

    Returns
    -------
    energies : np.array
        particle energies
    """
    energies = np.load('{}{:.2f}.npy'.format(save_directory, Ep))
    print('--- Loaded array from {}{:.2f}.npy ---'.format(save_directory, Ep))
    return energies

if __name__ == '__main__':
    Epmin = 18 # GeV
    Epmax = 700 # GeV
