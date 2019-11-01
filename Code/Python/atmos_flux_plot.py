# File: atmos_flux_plot.py
#
# Plotting the atmospheric pion and dark matter flux including three body decays
# Should be run as python atmos_dm_flux_plot.py [location=kcl or home] [particle=pion or dm]

import matplotlib.pyplot as plt
plt.style.use('ja')
import numpy as np
import seaborn as sns
import sys
import scipy.stats

from three_body import chi_energies
from pospelov import plot_pospelov
from atmos_dm_flux import kdemultifit

if __name__ == '__main__':
    # dEp = 1.0 # [GeV]
    # EpMin = Tmin(0.2)
    # EpMax = 18.0
    # nMC = 1
    # evals = proton_evals(dEp, EpMin, EpMax)
    # energies, weights = generate_energies_and_weights(evals, nMC)

    dEp = 5.0 # [GeV]
    EpMin = 10.0
    EpMax = 1000.0
    nMC = 500

    location = sys.argv[1]
    particle = sys.argv[2]
    if location not in ['kcl', 'home']:
        raise ValueError('Location not one of "kcl" or "home", please re run with one of these options.')
    if particle not in ['dm', 'pion']:
        raise ValueError('Particle not one of "pion" or "dm", please re reun with one of these options.')
    if location == 'kcl':
        save_directory = '/mnt/james/atmos/'
        save_directory = '/mnt/james/allMyStuff/BoostedDM/data/atmos'
    elif location == 'home':
        save_directory = '/Users/james/allMyStuff/BoostedDM/data/atmos'
    energies = np.load('{}energies.npy'.format(save_directory))
    weights = np.load('{}weights.npy'.format(save_directory))

    mchi = 0.001 # [GeV]
    mv = 0.01 # [GeV]
    mpi = 0.1349770 # [GeV]
    epsq = np.power(10.0, -5)
    branching_ratio = epsq*np.power(1 - np.power(mv/mpi, 2), 3)
    branching_ratio = np.power(10.0, -6)
    '''
    chienergies = np.array([])
    chiweights = np.array([])
    counter = 1
    for idx in range(0, len(energies)):
        new_kenergies = chi_energies(energies[idx], mpi, mv, mchi) - mchi
        chienergies = np.append(chienergies, new_kenergies)
        chiweights = np.append(chiweights, [weights[idx], weights[idx]])
        print('Completed {} out of {}'.format(counter, len(energies)), end='\r')
        counter += 1

    np.save('{}chienergies.npy'.format(save_directory), chienergies)
    np.save('{}chiweights.npy'.format(save_directory), chiweights)
    '''
    chienergies = np.load('{}chienergies.npy'.format(save_directory))
    chiweights = np.load('{}chiweights.npy'.format(save_directory))
    if particle == 'dm':
        energies = chienergies # FOR DM
        weights = chiweights # FOR DM

    logenergies = np.log10(energies)

    Nbins = int(np.ceil(np.sqrt(len(energies))))
    Nbins = 100
    _, bins = np.histogram(logenergies, bins=Nbins)

    if particle == 'dm':
        Ncuts = 5
        logcuts = np.linspace(-3.0, 2.0, Ncuts)
        cuts = np.power(10.0, logcuts)
        logbandwidths = np.linspace(-3.0, -1.0, Ncuts - 1)
        bandwidths = np.power(10.0, logbandwidths)
        kernel_evals, kernel_pdf = kdemultifit(energies, weights, cuts, bandwidths, branching_ratio, particle)
    elif particle == 'pion':
        Ncuts = 3
        logcuts = np.linspace(-1.0, 2.0, Ncuts)
        cuts = np.power(10.0, logcuts)
        logbandwidths = np.linspace(-2.0, -1.0, Ncuts - 1)
        bandwidths = np.power(10.0, logbandwidths)
        kernel_evals, kernel_pdf = kdemultifit(energies, weights, cuts, bandwidths, branching_ratio, particle)

    if particle == 'dm':
        weights = branching_ratio*energies*weights
    elif particle == 'pion':
        weights = energies*weights

    heights, plotbins = np.histogram(energies, bins=10**bins, weights=weights)
    binmidpointsarr = np.array([])
    binwidths = np.array([])
    for idx in range(0, len(heights)):
        midpoint = plotbins[idx] + 0.5*(plotbins[idx + 1] - plotbins[idx])
        binmidpointsarr = np.append(binmidpointsarr, midpoint)
        binwidths = np.append(binwidths, plotbins[idx + 1] - plotbins[idx])
    # We want to estimate the flux so we should divide by the bin width
    heights = heights*np.power(binwidths, -1.0)
    plt.rcParams['axes.facecolor'] = 'white'


    plt_color = '#FF9505'
    line_color = '#FB6107'
    kde_color = '#2D7DD2'

    # Pion Flux Plot
    if particle == 'pion':
        plt.figure(figsize=(7,7))
        plt.text(np.power(10.0, 0.0), np.power(10.0, -0.1), r'$n_{\textrm{\small MC}} = 500, \, E_p^{\textrm{\small max}} \sim 1\,\textrm{TeV}$', fontsize=10, ha="center", va="center", bbox=dict(boxstyle="round", lw=0.5, ec=(0.0, 0.0, 0.0), fc=(1.0, 1.0, 1.0)))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$E_\pi \, \textrm{[GeV]}$', fontsize=16)
        plt.ylabel(r'$E_\pi \textrm{d}\phi_\pi/\textrm{d}E_\pi \,\, \textrm{[cm}^{-2} \, \textrm{s}^{-1}\textrm{]}$', fontsize=16)
        plt.title('Atmospheric Pion Flux', fontsize=20)
        plt.bar(binmidpointsarr, heights, width=binwidths, bottom=np.power(10.0, -11), color='w', alpha=0.6, edgecolor=plt_color, linewidth=0.3, log=True, label='Pion Flux: $N_{\\textrm{\\tiny bins}} = $' + ' {}'.format(Nbins))
        plt.plot(binmidpointsarr, heights, color=line_color, linestyle='-', linewidth=1.0)
        plt.plot(kernel_evals, kernel_pdf, c=kde_color, linestyle='-', lw=1.0, label='Pion Flux: KDE Fit')
        plt.legend(frameon=False, fontsize=8)
        axes = plt.axis()
        plt.axis([np.power(10.0, -0.5), np.power(10.0, 2.0), np.power(10.0, -3.0), np.power(10.0, 0.1)])

        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.savefig('plots/atmos_pion_flux.pdf', bbox='tight')

    # Dark Matter Flux Plot
    if particle == 'dm':
        plt.figure(figsize=(7,7))
        plt.text(np.power(10.0, -1.5), np.power(10.0, -5.7), r'$\textrm{BR}(\pi^0 \rightarrow \gamma\chi\chi) = 10^{-6}, m_V = 0.01 \, \textrm{GeV}, m_\chi = 0.001 \, \textrm{GeV}$', fontsize=8, ha="center", va="center", bbox=dict(boxstyle="round", lw=0.5, ec=(0.0, 0.0, 0.0), fc=(1.0, 1.0, 1.0)))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$T_\chi \, \textrm{[GeV]}$', fontsize=16)
        plt.ylabel(r'$T_\chi \textrm{d}\phi_\chi/\textrm{d}T_\chi \,\, \textrm{[cm}^{-2} \, \textrm{s}^{-1}\textrm{]}$', fontsize=16)
        plt.title(r'Atmospheric Dark Matter Flux', fontsize=20)
        if location == 'kcl':
            plot_pospelov()
        elif location == 'home':
            plot_pospelov(save_directory='/Users/james/allMyStuff/BoostedDM/data/')
        plt.bar(binmidpointsarr, heights, width=binwidths, bottom=np.power(10.0, -11), color='w', alpha=0.6, edgecolor=plt_color, linewidth=0.3, log=True, label='DM Flux: $N_{\\textrm{\\tiny bins}} = $' + ' {}'.format(Nbins))
        plt.plot(binmidpointsarr, heights, color=line_color, linestyle='-', linewidth=1.0)
        plt.plot(kernel_evals, kernel_pdf, c=kde_color, linestyle='-', lw=0.7, label='DM Flux: KDE Fit')
        plt.legend(frameon=False, fontsize=8)
        axes = plt.axis()
        plt.axis([np.power(10.0, -3.0), np.power(10.0, 2), np.power(10.0, -10), np.power(10.0, -5.5)])

        plt.gca().set_xscale("log")
        plt.savefig('plots/atmos_dm_flux_pospelov.pdf', bbox='tight')
