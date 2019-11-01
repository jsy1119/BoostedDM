# File: atmos_mc.py
#
# Running the Monte Carlo simulations to generate the .npy files for large nMC for the atmospheric case

import numpy as np
from atmos_dm_flux import atmos_run_all_energies, atmos_generate_energies_and_weights
from pion_flux import proton_evals

if __name__ == '__main__':
    dEp = 5.0 # [GeV]
    EpMin = 10.0
    EpMax = 1000.0
    nMC = 50
    evals = proton_evals(dEp, EpMin, EpMax)
    print(evals)
    atmos_run_all_energies(evals, nMC)

    print('Finished MC stage, now generating energies and weights')
    energies, weights = atmos_generate_energies_and_weights(evals, nMC)

    save_directory = '/mnt/james/muon/'
    np.save('{}energies.npy'.format(save_directory), energies)
    np.save('{}weights.npy'.format(save_directory), weights)
    print('Finished run for dEp = {}, nMC = {}, max. proton energy = {} GeV'.format(dEp, nMC, EpMax))
