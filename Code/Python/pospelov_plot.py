# File: pospelov_plot.py
#
# for a given dark matter mass and cross section, plots the
# differential flux due to proton scattering as shown in Figure
# 1 of 1810.10543.

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
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from pospelov import dPxdTx, plot_pospelov
if __name__ == '__main__':
    Deff = 0.997 # kpc
    DeffCM = 3.086*np.power(10.0, 21)*Deff
    rho = 0.3 # GeV cm^-3
    sigma = np.power(10.0, -30.0) # cm^2

    mchiarr = np.array([0.01, 0.1, 1.0]) # GeV
    plotarr = []
    logTchiarr = np.linspace(-3.0, 0.0, 100)
    Tchiarr = np.power(10.0, logTchiarr)
    Tplargearr = np.array([50.0, 2.6, 1.5])
    line_colors = ['#0D0221', '#01295F', '#419D78']
    save_directory = '/mnt/james/allMyStuff/BoostedDM/data/'
    idx = 0
    '''
    for mchi in mchiarr:
        TdPdTarr = np.array([])
        for Tchi in Tchiarr:
            TdPdTarr = np.append(TdPdTarr, Tchi*dPxdTx(Tchi, DeffCM, rho, mchi, sigma, Tplarge=Tplargearr[idx]))
        np.save('{}pospelov{}.npy'.format(save_directory, mchi), TdPdTarr)
        plotarr.append(TdPdTarr)
        idx += 1
    '''
    '''
    for mchi in mchiarr:
        TdPdTarr = np.load('{}pospelov{}.npy'.format(save_directory, mchi))
        plotarr.append(TdPdTarr)
    '''
    plt.figure(figsize=(5,5))
    plt.title('Differential Flux (Pospelov)')
    plt.xlabel(r'$T_\chi$ [GeV]')
    plt.ylabel(r'$T_\chi \textrm{d}\Phi_\chi/\textrm{d}T_\chi \, \, \textrm{[cm}^{-2}\textrm{s}^{-1}\textrm{]}$')
    '''
    idx = 0
    for TdPdTarr in plotarr:
        plt.loglog(Tchiarr, plotarr[idx], lw=1.5, c=line_colors[idx], ls='-', label=r'$m_\chi =$ {} GeV'.format(mchiarr[idx]))
        idx += 1
    '''
    plot_pospelov()
    axes = plt.axis()
    plt.axis([np.power(10.0, -3.0), np.power(10.0, 0.0), np.power(10.0, -12.0), np.power(10.0, -4.0)])
    plt.legend()
    plt.savefig('plots/pospelov.pdf', bbox_inches="tight")
