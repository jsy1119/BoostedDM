# File: plotting.py
#
# Plotting testing for the pion flux

import matplotlib.pyplot as plt
plt.style.use('ja')
import numpy as np

from proton_flux import Tmin
from pion_flux import proton_evals, generate_energies_and_weights

if __name__ == '__main__':
    # dEp = 1.0 # [GeV]
    # EpMin = Tmin(0.2)
    # EpMax = 18.0
    # nMC = 1
    # evals = proton_evals(dEp, EpMin, EpMax)
    # energies, weights = generate_energies_and_weights(evals, nMC)

    dEp = 5.0 # [GeV]
    EpMin = 18.0 # no pions are produced in the simulations below this value
    EpMax = 1018.0
    nMC = 500

    save_directory = '/mnt/james/pion/'
    energies = np.load('{}energies.npy'.format(save_directory))
    weights = np.load('{}weights.npy'.format(save_directory))
    energies = 0.25*energies
    logenergies = np.log10(energies)
    Nbins = 75
    _, bins = np.histogram(logenergies, bins=Nbins)

    branching_ratio = np.power(10.0, -4)
    multiplicity = 2.0
    weights = multiplicity*branching_ratio*energies*weights # to plot E*flux

    plt.rcParams['axes.facecolor'] = 'white'

    plt_color = '#419D78'
    plt.figure()
    plt.text(np.power(10.0, 1.5), np.power(10.0, -9.2), r'$n_{\textrm{\small MC}} = 500, E_p^{\textrm{\small max}} \sim 1 \, \textrm{TeV}, \textrm{Br}(\pi^0 \rightarrow \gamma\chi\chi) = 10^{-4}$', fontsize=12, ha="center", va="center", bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0), fc=(1.0, 1.0, 1.0)))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$E_\chi \, \textrm{[GeV]}$', fontsize=16)
    plt.ylabel(r'$E_\chi \textrm{d}\phi_\chi/\textrm{d}E_\chi \,\, \textrm{[s}^{-1} \, \textrm{GeV}^{-1}\textrm{]}$', fontsize=16)
    plt.title('Dark Matter Flux', fontsize=20)
    plt.hist(energies, bins=10**bins, weights=weights, log=True, facecolor=plt_color, alpha=0.7, lw=0.2)
    #axes = plt.axis()
    #plt.axis([0, axes[1], axes[2], axes[3]])
    plt.gca().set_xscale("log")
    plt.savefig('plots/dm_flux.pdf')
