import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    int_energies = np.load('miguel/int_energies.npy')
    int_flux = np.load('miguel/int_flux.npy')
    atmos_energies = np.load('miguel/atmos_energies.npy')
    atmos_flux = np.load('miguel/atmos_flux.npy')

    pion_color = '#29AB87'

    plt.figure()
    fontsize = 14

    plt.text(np.power(10.0, -1.4), np.power(10.0, -13.0), r'$m_\chi = 0.001\,\mathrm{GeV}$', fontsize=fontsize)
    plt.text(np.power(10.0, -1.4), np.power(10.0, -13.8), r'$\sigma_\chi^{\mathrm{\small SI}} = 10^{-32}\,\mathrm{cm}^2$', fontsize=fontsize)
    plt.text(np.power(10.0, -1.4), np.power(10.0, -14.4), r'$\mathrm{BR}(\pi \rightarrow \chi\chi) = 10^{-6}$', fontsize=fontsize)

    plt.xlabel(r'$T_\chi\,\mathrm{[GeV]}$')
    plt.ylabel(r'$T_\chi\mathrm{d}\phi_\chi/\mathrm{d}T_\chi\,\mathrm{[cm}^{-2}\,\mathrm{s}^{-1}\mathrm{]}$')

    plt.plot(int_energies, int_flux, c=pion_color, ls='--', lw=1.5, label=r'Interstellar Flux')
    plt.plot(atmos_energies, atmos_flux, c=pion_color, lw=1.5, label=r'Atmospheric Flux')

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')

    axes = plt.axis()
    plt.axis([np.power(10.0, -3.0), np.power(10.0, 0.0), np.power(10.0, -15.0), np.power(10.0, -5.0)])

    ax = plt.gca()
    ax.tick_params(which='minor', length=2)

    plt.legend(loc='upper left', fontsize=14)
    plt.savefig('miguel/atmos_int.pdf')
