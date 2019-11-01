import matplotlib.pyplot as plt
plt.style.use('ja')
import numpy as np
import seaborn as sns
import sys
import scipy.stats

from atmos_dm_flux import atmos_run_all_energies, atmos_generate_energies_and_weights
from pion_flux import proton_evals

from three_body import chi_energies
from pospelov import plot_pospelov
from atmos_dm_flux import kdemultifit

if __name__ == '__main__':
    dEp = 1.0 # [GeV]
    EpMin = 10.0
    EpMax = 1000.0
    nMC = 5000
    evals = proton_evals(dEp, EpMin, EpMax)
    save_directory = '/mnt/james/muon/'
    energies = np.load('{}energies.npy'.format(save_directory))
    weights = np.load('{}weights.npy'.format(save_directory))

    logenergies = np.log10(energies)

    Nbins = int(np.ceil(np.sqrt(len(energies))))
    Nbins = 100
    _, bins = np.histogram(logenergies, bins=Nbins)

    Ncuts = 3
    logcuts = np.linspace(-1.0, 2.0, Ncuts)
    cuts = np.power(10.0, logcuts)
    logbandwidths = np.linspace(-2.0, -1.0, Ncuts - 1)
    bandwidths = np.power(10.0, logbandwidths)
    kernel_evals, kernel_pdf = kdemultifit(energies, weights, cuts, bandwidths, branching_ratio=1.0, particle='pion')
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

    plt.figure(figsize=(7,7))
    #plt.text(np.power(10.0, 0.0), np.power(10.0, -0.1), r'$n_{\textrm{\small MC}} = 500, \, E_p^{\textrm{\small max}} \sim 1\,\textrm{TeV}$', fontsize=10, ha="center", va="center", bbox=dict(boxstyle="round", lw=0.5, ec=(0.0, 0.0, 0.0), fc=(1.0, 1.0, 1.0)))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$E_\pi \, \textrm{[GeV]}$', fontsize=16)
    plt.ylabel(r'$E_\pi \textrm{d}\phi_\pi/\textrm{d}E_\pi \,\, \textrm{[cm}^{-2} \, \textrm{s}^{-1}\textrm{]}$', fontsize=16)
    plt.title('Atmospheric Muon Flux', fontsize=20)
    plt.bar(binmidpointsarr, heights, width=binwidths, bottom=np.power(10.0, -11), color='w', alpha=0.6, edgecolor=plt_color, linewidth=0.3, log=True, label='Muon Flux: $N_{\\textrm{\\tiny bins}} = $' + ' {}'.format(Nbins))
    plt.plot(binmidpointsarr, heights, color=line_color, linestyle='-', linewidth=1.0)
    plt.plot(kernel_evals, kernel_pdf, c=kde_color, linestyle='-', lw=1.0, label='Pion Flux: KDE Fit')
    plt.legend(frameon=False, fontsize=8)
    axes = plt.axis()
    plt.xlim(np.power(10.0, -0.5), np.power(10.0, 2.0))
    plt.ylim(np.power(10.0, -36.0), np.power(10.0, -22.0))

    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.savefig('plots/muons.pdf')
