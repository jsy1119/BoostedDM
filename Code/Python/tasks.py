from atmos_dm_flux import atmos_generate_energies_and_weights
from pion_flux import proton_evals
from three_body import chi_energies
from attenuation import Gdet
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.stats
import matplotlib
plt.rcParams['axes.linewidth'] = 1.75
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

if __name__ == '__main__':
    size = 100
    sigmaarray = np.logspace(-34.0, -28.0, size)
    # Gdetarr = np.empty(size)
    # for idx in range(0, size):
    #     Gdetarr[idx] = Gdet(sigmachi=sigmaarray[idx], mchi=0.001)
    # np.save('Gdetarr_mantle.npy', Gdetarr)
    # Gdetarr = np.load('Gdetarr_mantle.npy')
    #
    # GdetFun = interp1d(sigmaarray, Gdetarr, kind='slinear')
    # np.save('GdetFit.npy', np.array([GdetFun]))
    # sigmatest = np.logspace(-34.0, -28.0, 1000)

    #
    # fig, ax1 = plt.subplots()
    # ax1.set_ylabel(r'$G_{\textrm{\small det}}(\sigma_\chi^{\textrm{\small SI}})\,\textrm{[cm}^{-2}\textrm{]}$')
    # ax1.set_xlabel(r'$\sigma_\chi^{\textrm{\small SI}}\,\textrm{[cm}^2\textrm{]}$')
    #
    #
    # ax2 = ax1.twinx()
    # ax2.set_ylabel(r'$h_{\textrm{\small eff}}(\sigma_\chi^{\textrm{\small SI}})\,\textrm{[km]}$')
    # ax2.plot(sigmatest, heffarr)
    # ax1.semilogx(sigmaarray, Gdetarr, '+', color='#D1495B', markersize=0.0, markeredgewidth=0.5)
    # ax2.text(np.power(10.0, -30.6), 5.5, r'$m_\chi = 0.001\,\textrm{GeV}$', fontsize=10)
    # ax2.text(np.power(10.0, -30.6), 5.2, r'$n_{\textrm{\small eff}} = 5.05 \times 10^{19}\,\textrm{cm}^{-3}$', fontsize=10)
    # plt.title(r'The Effective Height')
    # plt.savefig('plots/gdet_mantle.pdf')





    dEp = 5.0 # [GeV]
    EpMin = 10.0
    EpMax = 1000.0
    nMC = 500
    evals = proton_evals(dEp, EpMin, EpMax)
    mpi = 0.134977
    mchi = 0.001
    mv = 0.01

    pion_save_directory = 'pion'
    eta_save_directory = 'eta'

    # pion_energies = np.load('{}energies.npy'.format(pion_save_directory))
    # pion_weights = np.load('{}weights.npy'.format(pion_save_directory))
    pion_chi_energies = np.load('{}chienergies.npy'.format(pion_save_directory))
    pion_chi_weights = np.load('{}chiweights.npy'.format(pion_save_directory))
    # eta_energies = np.load('{}energies.npy'.format(eta_save_directory))
    # eta_weights = np.load('{}weights.npy'.format(eta_save_directory))
    eta_chi_energies = np.load('{}chienergies.npy'.format(eta_save_directory))
    eta_chi_weights = np.load('{}chiweights.npy'.format(eta_save_directory))

    bw1 = np.power(10.0, -1.5)
    bw2 = np.power(10.0, -1.5)

    # pion_kernel = scipy.stats.gaussian_kde(dataset=pion_energies, weights=pion_weights, bw_method=bw1)
    # counts, _ = np.histogram(pion_energies, bins=1, weights=pion_weights)
    # pion_norm_factor = counts[0]

    pion_chi_kernel = scipy.stats.gaussian_kde(dataset=pion_chi_energies, weights=pion_chi_weights, bw_method=bw2)
    counts, _ = np.histogram(pion_chi_energies, bins=1, weights=pion_chi_weights)
    pion_chi_norm_factor = counts[0]

    # eta_kernel = scipy.stats.gaussian_kde(dataset=eta_energies, weights=eta_weights, bw_method=bw1)
    # counts, _ = np.histogram(eta_energies, bins=1, weights=eta_weights)
    # eta_norm_factor = counts[0]

    eta_chi_kernel = scipy.stats.gaussian_kde(dataset=eta_chi_energies, weights=eta_chi_weights, bw_method=bw2)
    counts, _ = np.histogram(eta_chi_energies, bins=1, weights=eta_chi_weights)
    eta_chi_norm_factor = counts[0]

    min1 = -1.0
    max1 = 3.0
    min2 = -4.0
    max2 = 3.0
    npts = 1000
    mpi = 0.134977
    meta = 0.547862
    Gdet = 2.529335739713634*np.power(10.0, 25)
    neff = 5.0*np.power(10.0, 19)
    heff = 5.0*np.power(10.0, 5)
    etabr = 5 * np.power(10.0, -3.0)
    pionbr = 6 * np.power(10.0, -4.0)

    # pion_kde_energies = np.logspace(min1, max1, npts)
    # pion_kde_flux = pion_norm_factor*pion_kernel.evaluate(pion_kde_energies)
    # import pandas as pd
    # pion_df = pd.DataFrame({'Epi': pion_kde_energies, 'dPhidE': pion_kde_flux})
    # pion_df.to_csv('pion_flux.dat', index=False)

    pion_chi_kde_energies = np.logspace(min2, max2, npts)
    pion_chi_kde_flux = pion_chi_norm_factor*pion_chi_kernel.evaluate(pion_chi_kde_energies)

    # eta_kde_energies = np.logspace(min1, max1, npts)
    # eta_kde_flux = eta_norm_factor*eta_kernel.evaluate(eta_kde_energies)

    eta_chi_kde_energies = np.logspace(min2, max2, npts)
    eta_chi_kde_flux = eta_chi_norm_factor*eta_chi_kernel.evaluate(eta_chi_kde_energies)

    pion_color = '#01295F'
    pion_color = '#29AB87'
    eta_color = '#D81159'
    pospelov_color = '#FF8514'
    plot_name = 'pion_eta_chi_no_title'

    from pospelov import dPxdTx
    import pandas as pd
    Deff = 0.997 # kpc
    DeffCM = 3.086*np.power(10.0, 21)*Deff
    rho = 0.3 # GeV cm^-3
    sigma = np.power(10.0, -32.0) # cm^2
    logTchiarr = np.linspace(-4.0, 0.5, 100)
    Tchiarr = np.power(10.0, logTchiarr)
    Tplarge = 100.0
    save_directory = '../data/'
    mchi = 0.001
    # dPdTarr = np.array([])
    # for Tchi in Tchiarr:
    #     dPdTarr = np.append(dPdTarr, dPxdTx(Tchi, DeffCM, rho, mchi, sigma, Tplarge=Tplarge))
    # np.save('{}pospelov{}.npy'.format(save_directory, mchi), dPdTarr)
    pospelov = np.load('{}pospelov{}.npy'.format(save_directory, mchi))
    pospelov_df = pd.read_csv('pospelov_flux.csv', header=None, names=['Tchi', 'TchidPhidTchi'])
    plt.figure()
    #plt.title(r'$\pi^0$ vs $\eta$ DM Fluxes')
    # plt.text(np.power(10.0, 1.5), np.power(10.0, -0.5), r'$n_{\textrm{\small eff}} = 5 \times 10^{19}\,\textrm{cm}^{-3}$', fontsize=10)
    # plt.text(np.power(10.0, 1.5), np.power(10.0, -0.8), r'$h_{\textrm{\small eff}} = 5\,\textrm{km}$', fontsize=10)
    fontsize = 12
    adjust = -4.3
    plt.text(7*np.power(10.0, -3), np.power(10.0, -4.0 + adjust), r'$m_\chi = 0.001\,\mathrm{GeV}$', fontsize=fontsize)
    plt.text(7*np.power(10.0, -3), np.power(10.0, -4.45 + adjust), r'$\sigma_\chi^{\mathrm{\small SI}} = 10^{-32}\,\mathrm{cm}^2$', fontsize=fontsize)
    plt.text(7*np.power(10.0, -3), np.power(10.0, -4.8 + adjust), r'$\mathrm{BR}(\eta \rightarrow \pi\chi\chi) = 5 \times 10^{-3}$', fontsize=fontsize)
    plt.text(7*np.power(10.0, -3), np.power(10.0, -5.2 + adjust), r'$\mathrm{BR}(\pi \rightarrow \gamma\chi\chi) = 6 \times 10^{-4}$', fontsize=fontsize)
    plt.xlabel(r'$T_\chi\,\mathrm{[GeV]}$')
    plt.ylabel(r'$T_\chi\mathrm{d}\phi_\chi/\mathrm{d}T_\chi\,\mathrm{[cm}^{-2}\,\mathrm{s}^{-1}\mathrm{]}$')
    #plt.plot(Tchiarr, Tchiarr*pospelov, c=pospelov_color, label=r'Elastic CRDM', linestyle='--')
    plt.plot(pospelov_df['Tchi'], pospelov_df['TchidPhidTchi'], c=pospelov_color, label=r'Elastic CRDM', linestyle='--')
    plt.plot(eta_chi_kde_energies, eta_chi_kde_energies*eta_chi_kde_flux*Gdet*etabr, c=eta_color, label=r'Inelastic CRDM ($\eta$)')
    plt.plot(pion_chi_kde_energies, pion_chi_kde_energies*pion_chi_kde_flux*Gdet*pionbr, c=pion_color, label=r'Inelastic CRDM ($\pi$)')
    np.save('miguel/atmos_energies.npy', pion_chi_kde_energies)
    np.save('miguel/atmos_flux.npy', pion_chi_kde_energies*pion_chi_kde_flux*Gdet*pionbr)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    axes = plt.axis()
    # plt.axis([np.power(10.0, min1), np.power(10.0, max1), np.power(10.0, -8.0), np.power(10.0, 0.5)])
    plt.axis([np.power(10.0, -4.0), np.power(10.0, 0.0), np.power(10.0, -10.0), np.power(10.0, -2.0)])
    ax = plt.gca()
    ax.tick_params(which='minor', length=3)
    plt.legend(loc='upper left', fontsize=14)
    plt.savefig('fluxes/{}.pdf'.format(plot_name))

    # energies = pion_energies - mpi
    # weights = pion_weights*neff*heff
    # kde_energies = pion_kde_energies - mpi
    # kde_flux = pion_kde_flux*neff*heff
    # plot_name = 'pion'
    #
    # logenergies = np.log(energies)
    #
    # Nbins = 300
    # _, logbins = np.histogram(logenergies, bins=Nbins, weights=weights)
    #
    #
    # heights, plotbins = np.histogram(energies, bins=10**logbins, weights=weights)
    # binmidpointsarr = np.array([])
    # binwidths = np.array([])
    # for idx in range(0, len(heights)):
    #     midpoint = plotbins[idx] + 0.5*(plotbins[idx + 1] - plotbins[idx])
    #     binmidpointsarr = np.append(binmidpointsarr, midpoint)
    #     binwidths = np.append(binwidths, plotbins[idx + 1] - plotbins[idx])
    # # We want to estimate the flux so we should divide by the bin width
    # heights = heights*np.power(binwidths, -1.0)
    #
    # plt_color = '#FF9505'
    # line_color = '#FB6107'
    # kde_color = '#2D7DD2'
    #
    # plt.figure()
    # plt.title(r'$\pi^0$ Flux')
    # plt.text(np.power(10.0, 1.5), np.power(10.0, -0.5), r'$n_{\textrm{\small eff}} = 5 \times 10^{19}\,\textrm{cm}^{-3}$', fontsize=10)
    # plt.text(np.power(10.0, 1.5), np.power(10.0, -0.8), r'$h_{\textrm{\small eff}} = 5\,\textrm{km}$', fontsize=10)
    # # plt.text(np.power(10.0, 0.8), np.power(10.0, -2.5), r'$\sigma_\chi^{\textrm{\small SI}} = 10^{-32}\,\textrm{cm}^2$', fontsize=10)
    # # plt.text(np.power(10.0, 0.8), np.power(10.0, -2.8), r'$\textrm{BR}(\pi \rightarrow \chi\chi) = 10^{-6}$', fontsize=10)
    # #plt.text(np.power(10.0, 1.5), np.power(10.0, -4.8), r'$\textrm{BR}(\pi^0 \rightarrow \chi\chi) = 10^{-6}$', fontsize=10)
    # plt.xlabel(r'$T_\pi\,\textrm{[GeV]}$')
    # plt.ylabel(r'$\textrm{d}\phi_\pi/\textrm{d}T_\pi$ [$\textrm{cm}^{-2}\,\textrm{s}^{-1}\,\textrm{GeV}^{-1}$]')
    # plt.plot(kde_energies, kde_flux, c=kde_color)
    # plt.bar(binmidpointsarr, heights, width=binwidths, bottom=np.power(10.0, -10), color='w', alpha=0.6, edgecolor=plt_color, linewidth=0.5, log=True)
    # plt.gca().set_xscale('log')
    # plt.gca().set_yscale('log')
    # axes = plt.axis()
    # plt.axis([np.power(10.0, min1), np.power(10.0, max1), np.power(10.0, -8.0), np.power(10.0, 0.5)])
    # #plt.axis([np.power(10.0, min2), np.power(10.0, max2), np.power(10.0, -10.0), np.power(10.0, -2.0)])
    # plt.savefig('fluxes/{}.pdf'.format(plot_name))


    # save_directory = '/mnt/james/atmos/'
    # energies = np.load('{}energies.npy'.format(save_directory))
    # weights = np.load('{}weights.npy'.format(save_directory))
    #
    # chienergies = np.array([])
    # chiweights = np.array([])
    # counter = 1
    # for idx in range(0, len(energies)):
    #     new_kenergies = chi_energies(energies[idx], mpi, mv, mchi) - mchi
    #     chienergies = np.append(chienergies, new_kenergies)
    #     chiweights = np.append(chiweights, [weights[idx], weights[idx]])
    #     print('Completed {} out of {}'.format(counter, len(energies)), end='\r')
    #     counter += 1
    #
    # np.save('{}chienergies.npy'.format(save_directory), chienergies)
    # np.save('{}chiweights.npy'.format(save_directory), chiweights)

    # from three_body import chi_energies
    # #EpiArr = np.array([0.134, 0.14, 0.16, 0.238, 0.3, 0.4, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0])
    # # for Epi in EpiArr:
    # size = 100000
    # energies = np.empty(size)
    # mchi = 0.001
    # mv = 0.01
    # mpi = 0.134
    # Epi = 0.1341
    # for idx in range(0, int(size/2)):
    #     new_es = chi_energies(Epi=Epi, mchi=mchi, mv=mv, mpi=mpi)
    #     energies[2*idx] = new_es[0]
    #     energies[2*idx + 1] = new_es[1]
    # Emax = energies.max()
    # print('Epi = {} GeV, EchiMax = {} GeV'.format(Epi, Emax))
    # plt.figure()
    # plt.hist(energies, density=True, bins=30, alpha=0.7)
    # plt.xlabel(r'$E_\chi\,\textrm{[GeV]}$')
    # plt.ylabel(r'P.D.F. $f(E_\chi)$')
    # plt.title(r'P.D.F. for $E_\pi$ = {:.3f} GeV'.format(Epi))
    # axes = plt.axis()
    # plt.plot([Epi, Epi], [0, axes[3]], 'k--', lw=1.0, c='#D1495B')
    # plt.plot([Epi/2, Epi/2], [0, axes[3]], 'k--', lw=1.0, c='#D1495B')
    # plt.axis([0.0, 1.05*Epi, 0.0, axes[3]])
    # plt.text(1.05*(Epi/2), .95*axes[3], r'$\frac{1}{2}E_\pi$', fontsize=10, verticalalignment='center')
    # plt.text(0.93*Epi, .95*axes[3], r'$E_\pi$', fontsize=10, verticalalignment='center')
    # plt.text(0.6*Epi, .8*axes[3], r'$E_\chi^{\textrm{\tiny max}}$' + ' = {:.3f} GeV'.format(Emax), fontsize=12)
    # plt.axis([0.0, 1.05*Epi, 0.0, axes[3]])
    # #plt.savefig('plots/chi_distr/{:.3f}.pdf'.format(Epi))
    # plt.show()
