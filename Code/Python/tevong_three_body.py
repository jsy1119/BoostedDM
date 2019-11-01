'''
Implements the three body decay kinematics as described in 1609.01770 for the decay Pi(0) -> V -> XX
'''

import numpy as np
import matplotlib.pyplot as plt
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


def lambdafn(a, b, c):
    r"""
    Parametrises the function :math:`\lambda(a, b, c) = a^2 + b^2 + c^2 - 2(ab + ac + bc)`.

    Parameters
    ----------
    a, b, c : float
        input values

    Returns
    -------
    lambda : float
        output of :math:`\lambda(a, b, c)`

    Examples
    --------
    >>> lambdafn(a=1.0, b=2.0, c=0.5)
    1.75
    """
    return np.abs(np.power(a, 2) + np.power(b, 2) + np.power(c, 2) - 2*(a*b + a*c + b*c))


def lorentz_boost(vvec, pvec):
    r"""
    Boosts the mediator 4-vector along the direction of the parent 4-vector using the standard parameterisation for Lorentz transformation matrices. Returns the boosted 4-vector.

    Parameters
    ----------
    vvec : np.array [len(vvec) = 4]
        mediator 4-vector
    pvec : np.array [len(pvec) = 4]
        pion 4-vector

    Returns
    -------
    outvec : np.array [len(outvec) = 4]
        boosted mediator 4-vector

    Examples
    --------
    >>> lorentz_boost(vvec=np.array([2.0, 0.0, 0.0, 1.0]), pvec=np.array([5.0, 2.0, 0.0, 0.0]))
    array([2.1821789 , 0.87287156, 0.        , 1.        ])
    """
    E1, p1x, p1y, p1z = pvec[0], pvec[1], pvec[2], pvec[3]
    E2, p2x, p2y, p2z = vvec[0], vvec[1], vvec[2], vvec[3]
    M1 = np.sqrt(np.abs(np.power(E1, 2) - np.power(p1x, 2) - np.power(p1y, 2) - np.power(p1z, 2)))
    g = E1/M1
    v = 1 - (1/np.power(g, 2))
    if v == 0:
        return vvec
    vx = p1x/E1
    vy = p1y/E1
    vz = p1z/E1
    L11, L12, L13, L14 = g, g*vx, g*vy, g*vz
    L21, L22, L23, L24 = g*vx, 1 + (g - 1)*np.power(vx, 2)/np.power(v, 2), (g - 1)*vx*vy/np.power(v, 2), (g - 1)*vx*vz/np.power(v, 2)
    L31, L32, L33, L34 = g*vy, (g - 1)*vy*vx/np.power(v, 2), 1 + (g - 1)*np.power(vy, 2)/np.power(v, 2), (g - 1)*vy*vz/np.power(v, 2)
    L41, L42, L43, L44 = g*vz, (g - 1)*vz*vx/np.power(v, 2), (g - 1)*vz*vy/np.power(v, 2), 1 + (g - 1)*np.power(vz, 2)/np.power(v, 2)

    E3 = L11*E2 + L12*p2x + L13*p2y + L14*p2z
    p3x = L21*E2 + L22*p2x + L23*p2y + L24*p2z
    p3y = L31*E2 + L32*p2x + L33*p2y + L34*p2z
    p3z = L41*E2 + L42*p2x + L43*p2y + L44*p2z

    return np.array([E3, p3x, p3y, p3z])


def cascade_pi_to_chi(Epi, Mchi, Mmed):
    r"""
    Carries out the cascade from a pion energy Epi to sample the possible dark matter energy distributions in the lab frame.

    Parameters
    ----------
    Epi : float
        energy of the pion [:math:`\textrm{GeV}`] (e.g. as output of crmc)
    mchi : float
        mass of the dark matter candidate [:math:`\textrm{GeV}`]
    Mmed : float
        mass of the mediator [:math:`\textrm{GeV}`] in the decay :math:`\pi_0 \rightarrow V \rightarrow \chi\chi`

    Returns
    -------
    chiarr : np.array
        length 2 array of sampled DM energies [:math:`\textrm{GeV}`]

    Examples
    --------
    >>> cascade_pi_to_chi(Epi=0.5, Mchi=0.001, Mmed=0.01)
    array([0.00755906, 0.03934015])
    """
    # Randomly sample the mediator decay plane
    ym = np.random.uniform(0.0, 1.0)
    zm = np.random.uniform(0.0, 1.0)
    thetam = np.arccos(-1 + 2*ym)
    phim = 2*np.pi*zm
    Mpi0 = 0.134977 # Neutral pion mass [GeV]
    lam = lambdafn(1, np.power(Mmed, 2)/np.power(Mpi0, 2), 0)

    # Construct mediator 4-momentum
    Em = (Mpi0/2.0)*(1.0 + np.power(Mmed, 2)/np.power(Mpi0, 2))
    pm = (Mpi0/2.0)*np.sqrt(np.abs(lam))

    pmx = pm*np.sin(thetam)*np.cos(phim)
    pmy = pm*np.sin(thetam)*np.sin(phim)
    pmz = pm*np.cos(thetam)

    # Boost mediator to lab frame
    [Em, pmx, pmy, pmz] = lorentz_boost(np.array([Em, pmx, pmy, pmz]), np.array([Epi, 0, 0, np.sqrt(np.abs(np.power(Epi, 2) - np.power(Mpi0, 2)))]))

    # Generate X momentum in mediator rest frame
    yd = np.random.uniform(0.0, 1.0)
    zd = np.random.uniform(0.0, 1.0)
    thetad = np.arccos(-1 + 2*yd)
    phid = 2*np.pi*zd
    lam = lambdafn(1, np.power(Mchi, 2)/np.power(Mmed, 2), np.power(Mchi, 2)/np.power(Mmed, 2))
    Ed = Mmed/2.0
    pd = (Mmed/2.0)*np.sqrt(np.abs(lam))
    pdx = pd*np.sin(thetad)*np.cos(phid)
    pdy = pd*np.sin(thetad)*np.sin(phid)
    pdz = pd*np.cos(thetad)
    Ed1, pd1x, pd1y, pd1z = Ed, pdx, pdy, pdz
    Ed2, pd2x, pd2y, pd2z = Ed, -pdx, -pdy, -pdz

    # Boost X to lab frame
    Ed1 = lorentz_boost(np.array([Ed1, pd1x, pd1y, pd1z]), np.array([Em, pmx, pmy, pmz]))[0]
    Ed2 = lorentz_boost(np.array([Ed2, pd2x, pd2y, pd2z]), np.array([Em, pmx, pmy, pmz]))[0]
    return np.array([np.abs(Ed1), np.abs(Ed2)])

if __name__ == '__main__':
    Epi0 = 5 # Gev
    Mchi = 0.1 # GeV
    Mmed = 1.0 # GeV
    chiEnergies = cascade_pi_to_chi(Epi0, Mchi, Mmed)
    print('Dark matter energies: {} GeV, {} GeV'.format(chiEnergies[0], chiEnergies[1]))
