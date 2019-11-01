# File: generate_atmos_flux_curves.py
#
# Generate the atmospheric flux curves (Tchi, Tchi dPhi/dTchi) across a range of dark matter masses.
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
    if location not in ['kcl', 'home']:
        raise ValueError('Location not one of "kcl" or "home", please re run with one of these options.')
    if location == 'kcl':
        save_directory = '/mnt/james/atmos/'
        #save_directory = '/mnt/james/allMyStuff/BoostedDM/data/atmos'
    elif location == 'home':
        save_directory = '/Users/james/allMyStuff/BoostedDM/data/atmos'
    energies = np.load('{}energies.npy'.format(save_directory))
    weights = np.load('{}weights.npy'.format(save_directory))

    mv = 0.1 # [GeV]
    mpi = 0.1349770 # [GeV]
    mchilogarr = np.linspace(-4.0, 0.95*np.log10(mv/2), 25)
    mchiarr = np.power(10.0, mchilogarr) # [GeV]

    epsq = np.power(10.0, -5)
    branching_ratio = epsq*np.power(1 - np.power(mv/mpi, 2), 3)
    branching_ratio = np.power(10.0, -6)
    count = 1
    for mchi in mchiarr:
        chienergies = np.array([])
        chiweights = np.array([])
        counter = 1

        # for idx in range(0, len(energies)):
        #     new_kenergies = chi_energies(energies[idx], mpi, mv, mchi) - mchi
        #     chienergies = np.append(chienergies, new_kenergies)
        #     chiweights = np.append(chiweights, [weights[idx], weights[idx]])
        #     print('Completed {} out of {} for mchi = {} GeV'.format(counter, len(energies), mchi), end='\r')
        #     counter += 1

        # np.save('{}{}chienergies{}.npy'.format(save_directory, mv, mchi), chienergies)
        # np.save('{}{}chiweights{}.npy'.format(save_directory, mv, mchi), chiweights)

        chienergies = np.load('{}{}chienergies{}.npy'.format(save_directory, mv, mchi))
        chiweights = np.load('{}{}chiweights{}.npy'.format(save_directory, mv, mchi))

        logchienergies = np.log10(chienergies)

        Nbins = int(np.ceil(np.sqrt(len(chienergies))))
        Nbins = 100
        _, bins = np.histogram(logchienergies, bins=Nbins)

        chiweights = branching_ratio*chienergies*chiweights

        heights, plotbins = np.histogram(chienergies, bins=10**bins, weights=chiweights)
        binmidpointsarr = np.array([])
        binwidths = np.array([])
        for idx in range(0, len(heights)):
            midpoint = plotbins[idx] + 0.5*(plotbins[idx + 1] - plotbins[idx])
            binmidpointsarr = np.append(binmidpointsarr, midpoint)
            binwidths = np.append(binwidths, plotbins[idx + 1] - plotbins[idx])
        # We want to estimate the flux so we should divide by the bin width
        heights = heights*np.power(binwidths, -1.0)
        np.save('{}{}binwidths{}.npy'.format(save_directory, mv, mchi), binwidths)
        np.save('{}{}binmidpoints{}.npy'.format(save_directory, mv, mchi), binmidpointsarr)
        np.save('{}{}fluxes{}.npy'.format(save_directory, mv, mchi), heights)
        print('\nCompleted {} out of {} dark matter masses.'.format(count, len(mchiarr)))
        count += 1
