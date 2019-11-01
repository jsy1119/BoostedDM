# File: full_mc.py
#
# Running the Monte Carlo simulations to generate the .npy files for large nMC

import numpy as np
from pion_flux import proton_evals, run_all_energies, generate_energies_and_weights
from proton_flux import Tmin

if __name__ == '__main__':
    dEp = 5.0 # [GeV]
    EpMin = 18.0 # no pions are produced in the simulations below this value
    EpMax = 1018.0
    nMC = 500
    evals = proton_evals(dEp, EpMin, EpMax)
    run_all_energies(evals, nMC)

    print('Finished MC stage, now generating energies and weights')
    energies, weights = generate_energies_and_weights(evals, nMC)

    save_directory = '/mnt/james/pion/'
    np.save('{}energies.npy'.format(save_directory), energies)
    np.save('{}weights.npy'.format(save_directory), weights)
    print('Finished run for dEp = {}, nMC = {}, max. proton energy = {} GeV'.format(dEp, nMC, EpMax))
