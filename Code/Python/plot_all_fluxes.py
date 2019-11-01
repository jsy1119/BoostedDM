# File: generate_atmos_flux_curves.py
#
# Generate the atmospheric flux curves (Tchi, Tchi dPhi/dTchi) across a range of dark matter masses.
# Should be run as python atmos_dm_flux_plot.py [location=kcl or home] [particle=pion or dm]

import matplotlib.pyplot as plt
import numpy as np
import sys
from pospelov import plot_pospelov
TICK_SIZE = 24
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)

if __name__ == '__main__':
    line_color = '#FB6107'
    grid_color = '#0D0221'
    location = sys.argv[1]
    if location not in ['kcl', 'home']:
        raise ValueError('Location not one of "kcl" or "home", please re run with one of these options.')
    if location == 'kcl':
        save_directory = '/mnt/james/atmos/'
        #save_directory = '/mnt/james/allMyStuff/BoostedDM/data/atmos'
    elif location == 'home':
        save_directory = '/Users/james/allMyStuff/BoostedDM/data/fluxes/'

    mv = 0.1 # [GeV]
    mpi = 0.1349770 # [GeV]
    mchilogarr = np.linspace(-4.0, 0.95*np.log10(mv/2), 25)
    mchiarr = np.power(10.0, mchilogarr) # [GeV]

    epsq = np.power(10.0, -5)
    branching_ratio = epsq*np.power(1 - np.power(mv/mpi, 2), 3)
    branching_ratio = np.power(10.0, -6)

    numrows = 5
    numcolumns = 5
    figwidth = 7*numcolumns
    figheight = 7*numrows
    fig = plt.figure(figsize=(figwidth, figheight))
    fig.suptitle(r"The atmospheric dark matter flux as a function of $m_\chi$"
    "\n"
    r"$\textrm{BR}(\pi^0 \rightarrow \gamma\chi\chi) = 10^{-6}, m_V = 0.1\,\textrm{GeV}$",
    fontsize=42, va='baseline')
    number = 1
    for mchi in mchiarr[:24]:
        heights = np.load('{}{}fluxes{}.npy'.format(save_directory, mv, mchi))
        binmidpointsarr = np.load('{}{}binmidpoints{}.npy'.format(save_directory, mv, mchi))
        ax = fig.add_subplot(numrows, numcolumns, number)
        plt.sca(ax)
        plt.title(r'$m_\chi =$ {:.3f} MeV'.format(mchi*1000), fontsize=24)

        if number in [1, 6, 11, 16, 21]:
            plt.ylabel(r'$T_\chi \textrm{d}\phi_\chi/\textrm{d}T_\chi \,\, \textrm{[cm}^{-2} \, \textrm{s}^{-1}\textrm{]}$', fontsize=28)
        if number in [21, 22, 23, 24, 20]:
            plt.xlabel(r'$T_\chi \, \textrm{[GeV]}$', fontsize=28)

        if location == 'kcl':
            plot_pospelov()
        elif location == 'home':
            plot_pospelov(save_directory='/Users/james/allMyStuff/BoostedDM/data/')
        plt.plot(binmidpointsarr, heights, color=line_color, linestyle='-', linewidth=2.0)
        plt.axis([np.power(10.0, -3.0), np.power(10.0, 2), np.power(10.0, -10), np.power(10.0, -5.5)])
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.grid(True, which='major', color=grid_color, alpha=0.6, linestyle='-.', linewidth=0.2)
        number += 1
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig('plots/{}all_fluxes.pdf'.format(mv), bbox='tight')
