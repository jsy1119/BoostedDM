# File: atmos_plot.py
#
# Plotting for the atmospheric proton fluxes

import numpy as np
import pandas as pd
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

from proton_flux import Fp_atmos, Rtilde, generate_Rtilde_arr, dPhiPatmosdT, Tmin
from crmc_utils import sigmapN
from air_density import suppression_factor
from atmos_dm_flux import geometrical_factor

if __name__ == '__main__':
    data_dir = '/home/k1893416/allMyStuff/BoostedDM/data/'
    plot_dir = '/home/k1893416/allMyStuff/BoostedDM/Python/plots/'
    line_color = '#DB504A'
    marker_color = '#254441'

    '''
    atmos_flux_df = pd.read_csv('{}atmos_flux.csv'.format(data_dir), sep=',', header=0)
    Rarr = np.array(atmos_flux_df['R'])
    Phiarr = np.power(10.0, -4)*np.array(atmos_flux_df['Phi'])[1:] # covert from m^-2 sr^-1 s^-1 GV^-1 to cm^-2 sr^-1 s^-1 GV^-1
    Rtildearr = generate_Rtilde_arr(Rarr)

    FpAarr = np.array([])
    # RAarr = np.array([])
    # for R in Rtildearr:
    #     if 10.0 <= R <= 1000.0:
    #         FpAarr = np.append(FpAarr, Fp_atmos(R))
    #         RAarr = np.append(RAarr, R)
    logRAarr = np.linspace(1.0, 3.0, 1000)
    RAarr = np.power(10.0, logRAarr)
    for R in RAarr:
        FpAarr = np.append(FpAarr, Fp_atmos(R))

    plt.figure(figsize=(5,5))
    plt.title('AMS Proton Flux')
    plt.xlabel(r'$R\,\textrm{[GV]}$')
    plt.ylabel(r'$\textrm{d}\Phi_p / \textrm{d}R \,\, \textrm{[cm}^{-2} \textrm{sr}^{-1}\textrm{s}^{-1}\textrm{GV}^{-1} \textrm{]}$')
    plt.loglog(RAarr, FpAarr, lw=1.0, c=line_color, ls='-', label=r'Fit down to 10 [GV]')
    plt.loglog(Rtildearr, Phiarr, marker='+', ms=4.0, mew=1.0, lw=0.0, c=marker_color)
    axes = plt.axis()
    plt.axis([1.0, axes[1], axes[2], axes[3]])
    plt.legend()
    plt.savefig('{}ams_flux.pdf'.format(plot_dir), bbox_inches="tight")
    '''

    # Plot differential flux as a function of energy
    '''
    dPhiPdTarr = np.array([])
    logEAarr = np.linspace(1.0, 2.8, 1000)
    EAarr = np.power(10.0, logEAarr)
    for E in EAarr:
        dPhiPdTarr = np.append(dPhiPdTarr, dPhiPatmosdT(E))

    plt.figure(figsize=(8,5))
    plt.title('AMS Proton Flux')
    plt.xlabel(r'$E_p\,\textrm{[GeV]}$')
    plt.ylabel(r'$E_p^{2.7}\textrm{d}\Phi_p / \textrm{d}E_p \,\, \textrm{[cm}^{-2} \textrm{s}^{-1}\textrm{GeV}^{-1} \textrm{]}$')
    plt.semilogx(EAarr, np.power(EAarr, 2.7)*dPhiPdTarr, lw=1.0, c=line_color, ls='-')
    axes = plt.axis()
    plt.axis([9.0, axes[1], axes[2], axes[3]])
    plt.savefig('{}ams_diff_flux.pdf'.format(plot_dir), bbox_inches="tight")
    '''
    # Plot pN cross section in energy range of interest.

    '''
    Epmin = 18 # GeV
    Epmax = 700 # GeV
    logEarr = np.linspace(np.log10(Epmin), np.log10(Epmax), 50)
    Earr = np.power(10.0, logEarr)
    sigmaarr = np.array([])
    counter = 1
    for E in Earr:
        print('Completed {} out of {} cross sections...'.format(counter, len(Earr)), end='\r')
        sigmaarr = np.append(sigmaarr, sigmapN(E))
        counter += 1

    plt.figure(figsize=(5,5))
    plt.title(r'$pN$ Cross Section')
    plt.xlabel(r'$E_p\,\textrm{[GeV]}$')
    plt.ylabel(r'$\sigma_{pN}^{\textrm{\small inel.}} \,\, \textrm{[cm}^{2}\textrm{]}$')
    plt.semilogx(Earr, sigmaarr, lw=1.5, c=line_color, ls='-', label='Inelastic Cross Section')
    axes = plt.axis()
    plt.legend(loc='best')
    plt.axis([9.0, axes[1], 2.5*np.power(10.0, -25), 2.8*np.power(10.0, -25)])
    plt.savefig('{}pNsigma.pdf'.format(plot_dir), bbox_inches="tight")
    '''

    # plot suppression factor for 0 < h < 180km
    # hmax = 180.0
    # sigmapN = 255*np.power(10.0, -27)
    # harr = np.linspace(0.0, hmax, 1000)
    '''
    Yarr = np.array([])
    for h in harr:
        Yarr = np.append(Yarr, suppression_factor(h, hmax))
    np.save('Yarr.npy', Yarr)
    '''
    # Yarr = np.load('Yarr.npy')
    #
    # plt.figure(figsize=(5,5))
    # plt.title('Atmospheric Supression Factor')
    # plt.xlabel(r'$h$ [km]')
    # plt.ylabel(r'$Y(h)$')
    # plt.plot(harr, Yarr, lw=1.5, c=line_color, ls='-', label='Suppression Factor')
    # axes = plt.axis()
    # plt.axis([0.0, hmax, 0.0, 1.05])
    # plt.legend(frameon=False)
    # plt.savefig('plots/suppression.pdf', bbox_inches="tight")

    h1 = 0.5
    npts = 100
    hmin = 10.0
    hmax = 180.0
    h2arr = np.linspace(hmin, hmax, 100)
    Fgarr = np.array([])
    counter = 1
    # for h2 in h2arr:
    #     Fgarr = np.append(Fgarr, geometrical_factor(h1, h2, npts))
    #     print('Completed {} out of {} factors'.format(counter, len(h2arr)), end='\r')
    #     counter += 1
    # np.save('Fgarr.npy', Fgarr)

    Fgarr = np.load('Fgarr.npy')

    plt.figure(figsize=(5,5))
    plt.title(r'Geometrical Factor')
    plt.xlabel(r'$H \, \textrm{[km]}$')
    plt.ylabel(r'$F_g(0.5\,\textrm{km}, H)\,\textrm{[cm}^{-2}\textrm{]}$')
    plt.plot(h2arr, Fgarr, lw=1.5, c=line_color, ls='-')
    axes = plt.axis()
    plt.axis([0.0, axes[1], 0.0, axes[3]])
    plt.savefig('plots/geometrical_factor.pdf', bbox_inches="tight")
