# File: three_body.py
#
# Implements the decay of pi0 -> V (on shell) -> XX

import numpy as np

def get_mediator_vector(Epi, mpi, mv):
    r"""
    Generates the 4-vector for the mediator V in the lab frame of the pion given a pion energy Epi and suitable masses assuming the pion is initially travelling in the positive z direction.

    Parameters
    ----------
    Epi : float
        pion energy [:math:`\textrm{GeV}`]
    mpi : float
        pion mass [:math:`\textrm{GeV}`] (0.134976 :math:`\textrm{GeV}`)
    mv : float
        mediator mass [:math:`\textrm{GeV}`] [NOTE: must be less than the pion mass else exception raised]

    Returns
    -------
    med_vector : np.array[4]
        mediator 4-vector in lab frame of pion :math:`[E_V, p_V^x, p_V^y, p_V^z]`

    Examples
    --------
    >>> get_mediator_vector(Epi=0.5, mpi=0.134976, mv=0.01)
    array([0.18517425, 0.05361832, 0.0358524 , 0.17328931])
    """
    try:
        # Check pion energy larger than mass
        if Epi < mpi:
            raise ValueError('ERROR: Pion energy smaller than the pion mass, insert a pion energy greater than {} GeV'.format(mpi))
        # Test if decay can be on shell
        if mpi < mv:
            raise ValueError('ERROR: Pion mass less than the mediator mass. The decay will not occur on shell, insert mediator mass less than {:.3f} GeV.'.format(mpi))

        pv = (np.power(mpi, 2) - np.power(mv, 2))/(2*mpi)

        # Sample theta and phi
        theta = np.arccos(-1 + 2*np.random.uniform(0, 1))
        phi = 2*np.pi*np.random.uniform(0, 1)

        # Compute gamma and beta
        g = Epi/mpi
        b = np.sqrt(1 - np.power(g, -2.0))

        # Construct 4-vector
        Ev = g*np.sqrt(np.power(pv, 2) + np.power(mv, 2)) + g*b*pv*np.cos(theta)
        pvx = pv*np.sin(theta)*np.cos(phi)
        pvy = pv*np.sin(theta)*np.sin(phi)
        pvz = g*b*np.sqrt(np.power(pv, 2) + np.power(mv, 2)) + g*pv*np.cos(theta)

        return np.array([Ev, pvx, pvy, pvz])

    except ValueError as ve:
        print(ve.args[0])

def get_chi_energies(med_vector, mpi, mv, mchi):
    r"""
    Generates the two energies for the decay :math:`V \rightarrow \chi\chi` in the lab frame of the original pion. The mediator vector should be generated using :py:func:`get_mediator_vector`.

    Parameters
    ----------
    med_vector : np.array[4]
        mediator 4-vector in the lab frame of pion :math:`[E_V, p_V^x, p_V^y, p_V^z]`
    mpi : float
        pion mass [:math:`\textrm{GeV}`] (0.134976 :math:`\textrm{GeV}`)
    mv : float
        mass of the mediator [:math:`\textrm{GeV}`]
    mchi : float
        mass of :math:`\chi` [:math:`\textrm{GeV}`]

    Returns
    -------
        energies : np.array[2] - array of two energies in the lab frame of the original pion [:math:`\textrm{GeV}`]

    Examples
    --------
    >>> get_chi_energies(med_vector=np.array([0.18517425, 0.05361832, 0.0358524 , 0.17328931]), mpi=0.134976, mv=0.01, mchi=0.001)
    array([0.18241224, 0.00276201])
    """
    try:
        # Check 4-vector length
        if len(med_vector) != 4:
            raise ValueError('ERROR: mediator 4-vector not of length 4.')
        # Check if decay on shell
        if mpi < mv:
            raise ValueError('ERROR: Pion mass less than the mediator mass. The decay will not occur on shell, insert mediator mass less than {:.3f} GeV.'.format(mpi))
        if mv < 2*mchi:
            raise ValueError('ERROR: mediator mass less than twice the dark matter mass. Resulting dark matter particles will not be on shell.')

        pchi = np.sqrt(0.25*np.power(mv, 2) - np.power(mchi, 2))
        Ev, pvx, pvy, pvz = med_vector
        qv = np.sqrt(np.power(pvx, 2) + np.power(pvy, 2) + np.power(pvz, 2))


        # Sample thetav and phiv
        thetav = np.arccos(-1 + 2*np.random.uniform(0, 1))
        phiv = 2*np.pi*np.random.uniform(0, 1)

        # Compute gamma and beta
        g = Ev/mv
        b = np.sqrt(1 - np.power(g, -2.0))

        # Compute energies by computing two terms individually and summing suitably
        t1 = g*np.sqrt(np.power(pchi, 2) + np.power(mchi, 2))
        t2 = (g*b*pchi/qv)*(pvx*np.sin(thetav)*np.cos(phiv) + pvy*np.sin(thetav)*np.sin(phiv) + pvz*np.cos(thetav))
        E1 = t1 + t2
        E2 = t1 - t2

        return np.array([E1, E2])

    except ValueError as ve:
        print(ve.args[0])

def chi_energies(Epi, mpi, mv, mchi):
    r"""
    Wrapper function to generate the two X energies for the decay pi0 -> V -> XX.

    Parameters
    ----------
    Epi : float
        pion energy [:math:`\textrm{GeV}`]
    mpi : float
        pion mass [:math:`\textrm{GeV}`] (0.134976 :math:`\textrm{GeV}`)
    mv : float
        mediator mass [:math:`\textrm{GeV}`]
    mchi : float
        mass of :math:`\chi` [:math:`\textrm{GeV}`]

    Returns
    -------
    energies : np.array[2]
        array of two energies in the lab frame of the original pion [:math:`\textrm{GeV}`]

    Notes
    -----
    To obtain the kinetic energies, subtract :math:`m_\chi` from the resulting values

    Examples
    --------
    >>> chi_energies(Epi=0.5, mpi=0.134976, mv=0.01, mchi=0.001)
    array([0.0550629 , 0.13542551])
    """
    med_vector = get_mediator_vector(Epi, mpi, mv)
    energies = get_chi_energies(med_vector, mpi, mv, mchi)
    return energies

if __name__ == '__main__':
    Epi = 10 # GeV
    mpi = 0.134976 # GeV
    mv = 0.01 # GeV
    mchi = 0.001 # GeV
    energies = chi_energies(Epi, mpi, mv, mchi)
    print('Dark Matter Energies: {:.3f} GeV and {:.3f} GeV'.format(energies[0], energies[1]))
