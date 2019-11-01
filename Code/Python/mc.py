# File: mc.py
#
# Running the Monte Carlo simulations to generate the .npy files

import numpy as np
from pion_flux import proton_evals, run_all_energies
from proton_flux import Tmin

if __name__ == '__main__':
    dEp = 1.0 # [GeV]
    EpMin = Tmin(0.2)
    EpMax = 1000.0
    nMC = 1
    evals = proton_evals(dEp, EpMin, EpMax)
    run_all_energies(evals, nMC)
