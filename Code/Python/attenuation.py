# File: attenuation.py
#
# implements the full attenuation of the dark matter and proton fluxes

import numpy as np
import matplotlib.pyplot as plt
from air_density import rho, suppression_factor
from mean_free_path import TzMin, mfp_nocore, TchiDenom, mfp_mantle
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
import warnings

def r(z, h, theta, RE=6378.1, hmax=180.0, hd=1.4):
    r"""
    Returns the radial distance used to compute the density along the line of sight for the protons coming from the top of the atmosphere.

    Parameters
    ----------
    z : float
        affine parameter, :math:`z`, to be integrated over along the line of sight [:math:`\textrm{km}`]
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hmax : float
        top of the atmosphere, where the AMS proton flux is known [:math:\textrm{km}]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)

    Returns
    -------
    r : float
        the radial distance, :math:`r` used to evaluate the density along the line of sight

    Notes
    -----
    We are implementing the following relationship,

    .. math:: r^2(z, h, \theta) = (R_E - h_d)^2 + z^2 - \textrm{sign}\left((R_E - h_d) - (R_E + h)\cos\theta\right) 2(R_E - h_d)z\left( 1 - \frac{(R_E + h)^2 \sin^2\theta}{\ell^2_d(h, \theta)} \right)^{1/2}

    where :math:`\ell_d(h, \theta)` is as defined in :py:func:`ld`.

    Examples
    --------
    >>> r(z=5.0, h=100.0, theta=0.0)
    6377.1
    >>> r(z=5.0, h=100.0, theta=np.pi/4)
    6370.280639328994
    >>> r(z=np.array([5.0, 10.0]), h=100.0, theta=0.0)
    array([6377.1, 6382.1])
    """
    cosa = np.sign((RE - hd) - (RE + h)*np.cos(theta))*np.power(1 - np.power((RE + h)*np.sin(theta), 2.0)*np.power(ld(h, theta), -2.0), 0.5)
    rsq = np.power(RE - hd, 2.0) + np.power(z, 2.0) - 2*(RE - hd)*z*cosa
    return np.power(rsq, 0.5)

def rp(zt, h, theta, RE=6378.1, hmax=180.0, hd=1.4):
    r"""
    Returns the radial distance used to compute the density along the line of sight for the protons coming from the top of the atmosphere.

    Parameters
    ----------
    zt : float
        affine parameter, :math:`\tilde{z}`, to be integrated over along the line of sight [:math:`\textrm{km}`]
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hmax : float
        top of the atmosphere, where the AMS proton flux is known [:math:\textrm{km}]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)

    Returns
    -------
    rp : float
        the radial distance, :math:`r_p` used to evaluate the density along the line of sight

    Notes
    -----
    We are implementing the following relationship,

    .. math:: r^2_p(\tilde{z}, h, \theta) = (R_E + h_{\textrm{max}})^2 + \tilde{z}^2 - 2(R_E + h_{\textrm{max}})\tilde{z}\left(1 - \frac{(R_E + h)^2(R_E - h_d)^2\sin^2\theta}{(R_E + h_{\textrm{max}})^2\ell^2_d(h, \theta)}\right)^{1/2}

    where :math:`\ell_d(h, \theta)` is as defined in :py:func:`ld`.

    Examples
    --------
    >>> rp(zt=5.0, h=100.0, theta=0.0)
    6553.1
    >>> rp(zt=5.0, h=100.0, theta=np.pi/4)
    6555.97346135917
    >>> rp(zt=np.array([5.0, 10.0]), h=100.0, theta=0.0)
    array([6553.1, 6548.1])
    """
    cosa = np.power(1 - np.power((RE + h)*(RE - hd)*np.sin(theta), 2.0)*np.power((RE + hmax)*ld(h, theta), -2.0), 0.5)
    rpsq = np.power((RE + hmax), 2) + np.power(zt, 2) - 2*(RE + hmax)*zt*cosa
    return np.power(rpsq, 0.5)

def zstar(h, theta, RE=6378.1, hd=1.4):
    r"""
    Returns the line of sight distance, but only inside the Earth after dark matter produced at a point :math:`(h, \theta)`.

    Parameters
    ----------
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)

    Returns
    -------
    zstar : float
        line of sight distance between Earth entry point and detector [:math:`\textrm{km}`]

    Notes
    -----
    We are implementing the equation,

    .. math:: z_\star = \frac{1}{2}\left(b(h, \theta) + \sqrt{b^2(h, \theta) + 4\left(R_E^2 - (R_E - h_d)^2\right)}\right)

    where,

    .. math:: b(h, \theta) = \textrm{sign}\left((R_E - h_d) - (R_E + h)\cos\theta\right)\cdot 2(R_E - h_d)\left(1 - \frac{(R_E + h)^2 \sin^2\theta}{\ell_d^2(h, \theta)}\right)^{1/2}

    Examples
    --------
    >>> zstar(h=100.0, theta=0.0, hd=1.4)
    """
    b = np.sign((RE - hd) - (RE + h)*np.cos(theta))*np.power(1 - np.power((RE + h)*np.sin(theta), 2.0)*np.power(ld(h, theta), -2.0), 0.5)*2*(RE - hd)
    zstar = 0.5*(b + np.sqrt(np.power(b, 2.0) + 4*(np.power(RE, 2.0) - np.power(RE - hd, 2.0))))
    return zstar

def ld(h, theta, RE=6378.1, hd=1.4):
    r"""
    Returns the line of sight distance between the dark matter production point and the detector.

    Parameters
    ----------
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)

    Returns
    -------
    ld : float
        line of sight distance between production point and detector [:math:`\textrm{km}`]

    Notes
    -----
    We are implementing the equation,

    .. math:: \ell^2_d(h, \theta) = (R_E + h)^2 + (R_E - h_d)^2 - 2(R_E + h)(R_E - h_d)\cos\theta

    Examples
    --------
    >>> ld(h=100.0, theta=0.0)
    106.0
    >>> ld(h=10.0, theta=np.pi/4)
    4883.1395076078725
    >>> ld(h=np.array([10.0, 20.0, 30.0]), theta=np.pi/4)
    array([4883.13950761, 4887.0030027 , 4890.88389209])
    """
    ldsq = np.power(RE + h, 2) + np.power(RE - hd, 2) - 2*(RE + h)*(RE - hd)*np.cos(theta)
    return np.power(ldsq, 0.5)

def lp(h, theta, RE=6378.1, hmax=180.0, hd=1.4):
    r"""
    Returns the line of sight distance between the top of the atmosphere and the dark matter production point.

    Parameters
    ----------
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hmax : float
        top of the atmosphere, where the AMS proton flux is known [:math:\textrm{km}]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)

    Returns
    -------
    lp : float
        line of sight distance between top of the atmosphere and production point [:math:`\textrm{km}`]

    Notes
    -----
    We are implementing

    .. math:: \ell_p(h, \theta) = (R_E + h_{\textrm{max}})\left(1 - \frac{(R_E + h)^2(R_E - h_d)^2 \sin^2\theta}{(R_E + h_{\textrm{max}})^2 \ell^2_d(h, \theta)}\right)^{1/2} - (R_E + h)\left(1 - \frac{(R_E - h_d)^2 \sin^2\theta}{\ell^2_d(h, \theta)}\right)^{1/2}

    Examples
    --------
    >>> lp(h=100.0, theta=0.0)
    80.0
    >>> lp(h=100.0, theta=0.5)
    268.9261976557591
    """
    lp = (RE + hmax)*np.power(1 - np.power(RE + h, 2)*np.power(RE - hd, 2)*np.power(np.sin(theta), 2)*np.power(RE + hmax, -2.0)*np.power(ld(h, theta), -2.0), 0.5) - (RE + h)*np.power(1 - np.power(RE - hd, 2)*np.power(np.sin(theta), 2)*np.power(ld(h, theta), -2.0), 0.5)
    return lp

def rhoElos(z, h, theta, RE=6378.1, hmax=180.0, hd=1.4):
    r"""
    Returns the earth density as a function of the affine parameter :math:`z` along the line of sight.

    Parameters
    ----------
    z : float
        affine parameter, :math:`z`, to be integrated over along the line of sight [:math:`\textrm{km}`]
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hmax : float
        top of the atmosphere, where the AMS proton flux is known [:math:\textrm{km}]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)

    Returns
    -------
    rhoElos : float
        number density of Earth a distance :math:`z` along the line of sight

    Notes
    -----
    We are essentially computing :math:`n_E\left(r(z, h, \theta)\right)` along the line of sight from :math:`z = 0` to :math:`z = \ell_d(h, \theta)`

    Examples
    --------
    >>> rhoElos(z=5.0, h=100.0, theta=0.0)
    array([0.])
    >>> rhoElos(z=np.array([0.0, 5.0, 10.0]), h=100.0, theta=np.pi/4)
    array([0.00000000e+00, 5.16731583e+22, 5.16731583e+22])
    """
    radius = r(z, h, theta, RE, hmax, hd)
    return rhoE(radius)

def rhoN(zt, h, theta, RE=6378.1, hmax=180.0, hd=1.4):
    r"""
    Returns the air density as a function of the affine parameter :math:`\tilde{z}` along the line of sight.

    Parameters
    ----------
    zt : float
        affine parameter, :math:`\tilde{z}`, to be integrated over along the line of sight [:math:`\textrm{km}`]
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hmax : float
        top of the atmosphere, where the AMS proton flux is known [:math:\textrm{km}]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)

    Returns
    -------
    rhoN : float
        number density of nitrogen a distance :math:`\tilde{z}` along the line of sight

    Notes
    -----
    We are essentially computing :math:`n_N\left(r_p(\tilde{z}, h, \theta) - R_E\right)` along the line of sight from :math:`\tilde{z} = 0` to :math:`\tilde{z} = \ell_p(h, \theta)`

    Examples
    --------
    >>> rhoN(zt=5.0, h=100.0, theta=0.0)
    27346504819.10922
    >>> rhoN(zt=np.array([0.0, 5.0, 10.0]), h=100.0, theta=np.pi/4)
    array([2.36324087e+10, 2.48066000e+10, 2.65849888e+10])
    """
    height = rp(zt, h, theta, RE, hmax, hd) - RE
    return rho(height)

def Yd(h, theta, sigmachi, Asq=165.025, RE=6378.1, hmax=180.0, hd=1.4):
    r"""
    Returns the suppression factor :math:`Y_p(h, \theta)` for a given height and angle.

    Parameters
    ----------
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    sigmachi : float
        spin-independent WIMP cross section, :math:`\sigma^{\textrm{SI}}_\chi` [:math:`\textrm{cm}^2`]
    Asq : float
        coherence factor by which the spin-independent cross section increases by, :math:`\sigma_{\chi N} = \sigma_\chi^{\textrm{SI}} A^2`
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hmax : float
        top of the atmosphere, where the AMS proton flux is known [:math:\textrm{km}]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)

    Returns
    -------
    Yd : float
        the suppression factor :math:`Y_d(h, \theta, \sigma_\chi^{\textrm{SI}})`

    Notes
    -----
    To compute the suppression factor, we are performing the integral,

    .. math:: Y_d(h, \theta) = \exp\left(-\sigma_{\chi N}\int_{0}^{\ell_d(h, \theta)}{\textrm{d}z\,n_E\left(r(z)\right)}\right)

    Examples
    --------
    >>> Yd(h=10.0, theta=np.pi/32, sigmachi=np.power(10.0, -32))
    0.20690033335769029
    """
    length = ld(h, theta, RE, hd)
    integral = quad(rhoElos, a=0.0, b=length, args=(h, theta, RE, hmax, hd))[0]
    Yd = np.exp(-Asq*sigmachi*integral*np.power(10.0, 5))
    return Yd

def Yp(h, theta, RE=6378.1, hmax=180.0, hd=1.4, sigmapN=255*np.power(10.0, -27)):
    r"""
    Returns the suppression factor :math:`Y_p(h, \theta)` for a given height and angle.

    Parameters
    ----------
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hmax : float
        top of the atmosphere, where the AMS proton flux is known [:math:\textrm{km}]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)
    sigmapN : float
        pN cross section [:math:`\textrm{cm}^2`]

    Returns
    -------
    Yp : float
        the suppression factor :math:`Y_p(h, \theta)`

    Notes
    -----
    To compute the suppression factor, we are performing the integral,

    .. math:: Y_p(h, \theta) = \exp\left(-\sigma_{pN}^{\textrm{inel.}}\int_{0}^{\ell_p(h, \theta)}{\textrm{d}\tilde{z}\,n_N\left(r_p(\tilde{z}) - R_E\right)}\right)

    Examples
    --------
    >>> Yp(h=10.0, theta=0.0)
    0.04505642694802813
    """
    length = lp(h, theta, RE, hmax, hd)
    integral = quad(rhoN, a=0.0, b=length, args=(h, theta, RE, hmax, hd))[0]
    Yp = np.exp(-sigmapN*integral*np.power(10.0, 5))
    return Yp

def generate_Yp_interpolator(hmax=180.0, thetamax=np.pi, npts=10000, hd=1.4, sigmapN=255*np.power(10.0, -27), savename='YpInterpolation.npy'):
    r"""
    Generates a scipy interpolation object which is then saved to the file `savename`. This can then be loaded using :py:func:`load_Yp_interpolator` and called as a normal function. This is to increase the performance speed for the integration in :py:func:`Gdet`.

    Parameters
    ----------
    hmax : float
        top of the atmosphere, where the AMS proton flux is known [:math:\textrm{km}]
    thetamax : float
        maximum angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    npts : int
        number of sampling points for :math:`h` and :math:`\theta` to perform the interpolation on
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)
    sigmapN : float
        pN cross section [:math:`\textrm{cm}^2`]
    savename : str
        save location for the interpolation object, can be loaded using :py:func:`load_Yp_interpolator`.

    Notes
    -----
    This makes use of the `scipy.interpolate.interp2d` class. For more information, see the documentation `here <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.interp2d.html>`_.
    """
    harr = np.linspace(0.0, hmax, npts)
    thetaarr = np.linspace(0.0, thetamax, npts)
    H, T = np.meshgrid(harr, thetaarr)
    Yparr = np.empty([npts, npts])
    count = 1
    for i in range(0, npts):
        for j in range(0, npts):
            Yparr[i][j] = Yp(h=H[i][j], theta=T[i][j], hd=hd, sigmapN=sigmapN)
            print('Completed {} out of {} evaluations'.format(count, np.power(npts, 2)), end='\r')
            count += 1
    YpFun = interp2d(harr, thetaarr, Yparr)
    np.save(savename, YpFun)

def load_Yp_interpolator(savename='YpInterpolation.npy'):
    r"""
    After creating an interpolation object using :py:func:`generate_Yp_interpolator`, loads the object from the saved file. This can then be called as a normal function.

    Parameters
    ----------
    savename : str
        save location for the interpolation object

    Returns
    -------
    YpFun : scipy.interpolate.interpolate.interp2d
        interpolation function object

    Examples
    --------
    >>> YpFun = load_Yp_interpolator('YpInterpolation.npy')
    >>> H, T = np.meshgrid(np.linspace(0.0, 180.0, 1000), np.linspace(0.0, np.pi, 1000))
    >>> Yp = YpFun(H, T) # returns array of size (1000, 1000)
    """
    obj = np.load(savename)
    YpFun = obj.item()
    return YpFun

def rhoE(r, RE=6378.1, mavg=11.8871, Asq=165.025):
    r"""
    Returns the density of the Earth at a radius :math:`r < R_E`. If the radius is greater than this value, the density is assumed to be zero. The parametrisation is from `Preliminary reference Earth model <https://www.cfa.harvard.edu/~lzeng/papers/PREM.pdf>`_ on Page 308.

    Parameters
    ----------
    r : np.array
        radius at which the density is measured [:math:`\textrm{km}`]
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    mavg : float
        average atomic mass of Earth constituent elements [atomic units]

    Returns
    -------
    rhoE : np.array
        the number density of the Earth at a radius :math:`r\,\textrm{km}` [:math:`\textrm{cm}^{-3}`]

    Examples
    --------
    >>> rhoE(1500.0)
    array([6.08420051e+23])
    >>> rhoE(np.array([1500.0, 2500.0]))
    array([6.08420051e+23, 5.66919002e+23])
    """
    if type(r) != np.ndarray:
        r = np.array([r])
    rhoE = np.empty(len(r))
    a = 6371.0 # scaling parameter [km]
    # Inner Core
    mask = (0 <= r) & (r < 1221.5)
    rhoE[mask] = 13.0885 - 8.8381*np.power(r[mask]/a, 2)
    # Outer Core
    mask = (1221.5 <= r) & (r < 3480.0)
    rhoE[mask] = 12.5815 - 1.2638*r[mask]/a - 3.6426*np.power(r[mask]/a, 2) - 5.5281*np.power(r[mask]/a, 3)
    # Lower Mantle
    mask = (3480.0 <= r) & (r < 5701.0)
    rhoE[mask] = 7.9565 - 6.4761*r[mask]/a + 5.5283*np.power(r[mask]/a, 2) - 3.0807*np.power(r[mask]/a, 3)
    # Transition Zone
    mask = (5701.0 <= r) & (r < 5771.0)
    rhoE[mask] = 5.3197 - 1.4836*r[mask]/a
    mask = (5771.0 <= r) & (r < 5971.0)
    rhoE[mask] = 11.2494 - 8.0298*r[mask]/a
    mask = (5971.0 <= r) & (r < 6151.0)
    rhoE[mask] = 7.1089 - 3.8045*r[mask]/a
    # LVZ + LID
    mask = (6151.0 <= r) & (r < 6346.6)
    rhoE[mask] = 2.6910 + 0.6924*r[mask]/a
    # Crust
    mask = (6346.6 <= r) & (r < 6356.0)
    rhoE[mask] = 2.900
    mask = (6356.0 <= r) & (r < 6368.0)
    rhoE[mask] = 2.600
    # Ocean
    mask = (6368.0 <= r) & (r < 6371.0)
    rhoE[mask] = 1.020
    # Above
    mask = (6371.0 <= r)
    rhoE[mask] = 0.0

    # Convert to cm^-3
    NA = 6.022*np.power(10.0, 23)
    rhoE = rhoE*NA/mavg
    return rhoE

def Gdet_integrand(h, theta, RE=6378.1, hmax=180.0, hd=1.4, sigmapN=255*np.power(10.0, -27)):
    r"""
    Returns the integrand to be integrated over the :math:`(h, \theta)` region in :py:func:`Gdet`.

    Parameters
    ----------
    h : float
        height above the surface of the Earth [:math:`\textrm{km}`]
    theta : float
        angle, :math:`\theta` between the detector at a depth :math:`h_d` below the Earth's surface, the centre of the Earth, and the dark matter production point
    RE : float
        radius of the Earth [:math:`R_E = 6378.1\,\textrm{km}`]
    hmax : float
        top of the atmosphere, where the AMS proton flux is known [:math:\textrm{km}]
    hd : float
        depth of the detector (for Xenon 1T, :math:`h_d = 1.4\,\textrm{km}`)
    sigmapN : float
        pN cross section [:math:`\textrm{cm}^2`]

    Returns
    -------
    Gdet_integrand : float
        integrand [:math:`\textrm{cm}^{-3}`]

    Notes
    -----
    We are computing the integrand,

    .. math:: 2\pi(R_E + h)^2 \sin\theta\frac{Y_d(h, \theta; \sigma^{\textrm{SI}}_{\chi})Y_p(h, \theta)}{\ell_d^2(h, \theta)} n_N(h)

    Examples
    --------
    >>> Gdet_integrand(h=15.0, theta=0.1)
    13736347101234.008
    """
    return np.power(RE + h, 2.0)*np.sin(theta)*rho(h)*Yp(h, theta)*np.power(ld(h, theta), -2.0)

def Gdet(sigmachi, mchi, dt=np.power(10.0, -5.0), dh=np.power(10.0, -0.5)):
    r"""
    Returns the geometrical factor :math:`G_{\textrm{det}}(\sigma_\chi^{\textrm{SI}})` that multiplies the flux [:math:`\textrm{sr}\,\textrm{cm}^{-2}`].

    Parameters
    ----------
    sigmachi : float
        the spin-independent WIMP-nucleon cross-section, :math:`\sigma_\chi^{\textrm{SI}}` [:math:`\textrm{cm}^{2}`]

    mchi : float
        dark matter particle mass [:math:`\textrm{GeV}`]

    Returns
    -------
    Gdet : float
        the geometrical factor [:math:`\textrm{sr}\,\textrm{cm}^{-2}`]

    Notes
    -----
    This assumes that all other parameters contained in the internal function definitions take their default values e.g. :math:`R_E = 6378.1 \, \textrm{km}`. The quantity being calculated is the following;

    .. math:: G_{\textrm{det}}(\sigma^{\textrm{SI}}_{\chi}) = \int_0^{h_{\textrm{ max}}}{\textrm{d}h\,(R_E + h)^2 \int_{0}^{2\pi}{\textrm{d}\phi\, \int_{-1}^{+1}{\textrm{d}\cos\theta\, \frac{Y_d(h, \theta; \sigma^{\textrm{SI}}_{\chi})Y_p(h, \theta)}{2\pi\ell_d^2(h, \theta)} n_N(h)  }  }  }

    Furthermore, you should have run :py:func:`generate_Yp_interpolator` saving to a filename `YpInterpolation.npy` in the current working directory.

    Examples
    --------
    >>> Gdet(sigmachi=np.power(10.0, -32), mchi=0.001, dt=np.power(10.0, -5), dh=np.power(10.0, -0.5))
    """
    RE = 6378.1 # km
    # Ypfun = load_Yp_interpolator('YpInterpolation.npy')
    # print('Loaded Yp Function')
    theta_max_fun = generate_theta_max_interpolation(sigmachi=sigmachi, mchi=mchi)
    harr = np.arange(0.0, 180.0, dh)

    integrand = 0.0
    count = 1
    for h in harr:
        tmax = theta_max_fun(h)
        if tmax != 0.0:
            tarr = np.arange(0.0, tmax, dt)
        else:
            tarr = np.array([0.0])
        htemparr = np.full(len(tarr), h)
        farr = np.power(RE + h, 2.0)*np.sin(tarr)*rho(h)*suppression_factor(h)*np.power(ld(htemparr, tarr), -2.0)
        integrand += farr.sum()*dt*dh
        print('Completed {} out of {} heights'.format(count, len(harr)), end='\r')
        count += 1
    # convert to cm
    integrand = integrand*np.power(10.0, 5)
    print('\n--------\nResults:\n--------\n\nIntegral: {}, sigmachi = {}, mchi = {}\n'.format(integrand, sigmachi, mchi))
    return integrand


def generate_theta_max_interpolation(sigmachi, mchi):
    r"""
    Returns the interpolated function for the contour in the :math:`(h, \theta)` plane defining the integration region. This is called in :py:func:`Gdet`.

    Parameters
    ----------
    sigmachi : float
        the spin independnent WIMP-Nucleon cross section, :math:`\sigma_\chi^{\textrm{SI}}` [:math:`\textrm{cm}^2`]

    Returns
    -------
    theta_max_fun : scipy.interpolate.interp1d
        the interpolated function which can then be called as `theta_max_fun(h)`

    Notes
    -----
    This makes use of the `scipy.interpolate.interp1d` function. See `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d>`_ for more information.

    Examples
    --------
    >>> theta_max_fun = theta_max_fun(sigmachi=np.power(10.0, -32))
    >>> harr = np.linspace(0.0, 180.0, 100)
    >>> theta_max_fun(harr)
    """
    warnings.filterwarnings('error')
    RE = 6378.1
    H, T = np.meshgrid(np.linspace(0.0, 180.0, 1000), np.linspace(0.0, np.pi, 1000))
    Z = zstar(H, T)
    testzarr = np.linspace(0.0, 2*RE, 100000)
    mfp = mfp_mantle(sigmachi, mchi)
    denomarr = TchiDenom(Tz=TzMin(), l=mfp, z=testzarr, mchi=mchi)
    mask = (denomarr > 0)
    zmax = testzarr[mask].max()
    try:
        CS = plt.contour(H, T, Z, levels=[zmax])
        h, t = CS.allsegs[0][0].T
        theta_max_fun = interp1d(h, t)
        return theta_max_fun
    except Warning:
        h = np.linspace(0.0, 180.0, 1000)
        if sigmachi > np.power(10.0, -30):
            t = np.full(1000, 0.0)
        else:
            t = np.full(1000, np.pi)
        theta_max_fun = interp1d(h, t)
        return theta_max_fun

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    # Gdet(sigmachi=np.power(10.0, -32), mchi=0.001)
    # print('Gdet() = {}'.format(G))
    '''
    dist = sys.argv[1]
    plt.figure()
    size = 1000
    harr, thetaarr = np.meshgrid(np.linspace(0, 180, size), np.linspace(0, np.pi, size))
    if dist == 'lp':
        larr = lp(h=harr, theta=thetaarr)
    elif dist == 'ld':
        larr = ld(h=harr, theta=thetaarr)
    plt.contourf(harr, thetaarr, larr)
    plt.xlabel(r'$h \, \textrm{[km]}$')
    plt.ylabel(r'$\theta$')
    if dist == 'lp':
        plt.colorbar(label=r'$\ell_p(h, \theta)\, \textrm{[km]}$')
        plt.title(r'Contour plot of $\ell_p(h, \theta)$')
        plt.savefig('plots/lpcontour.pdf', bbox='tight')
    elif dist == 'ld':
        plt.colorbar(label=r'$\ell_d(h, \theta)\, \textrm{[km]}$')
        plt.title(r'Contour plot of $\ell_d(h, \theta)$')
        plt.savefig('plots/ldcontour.pdf', bbox='tight')

    if dist == 'lp':
        plt.figure()
        htest = np.linspace(0.0, 180.0, 100)
        thetaarr = np.array([np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2])
        labels = np.array([r'$\pi/32$', r'$\pi/16$', r'$\pi/8$', r'$\pi/4$', r'$\pi/2$'])
        colors = np.array(['#01295F', '#419D78', '#ECBA82', '#D1495B', '#98DFEA'])
        idx = 0
        for theta in thetaarr:
            lparr = lp(htest, theta)
            plt.plot(htest, lparr, label=r'$\theta = \,$' + labels[idx])
            idx += 1
        plt.ylabel(r'$\ell_p(h, \theta)\,\textrm{[km]}$')
        plt.xlabel(r'$h\,\textrm{[km]}$')
        plt.legend()
        plt.title(r'$\ell_p(h, \theta)$ for different values of $\theta$')
        plt.savefig('plots/lpexample.pdf', bbox='tight')

    if dist == 'ld':
        plt.figure()
        htest = np.linspace(0.0, 180.0, 100)
        thetaarr = np.array([np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2])
        labels = np.array([r'$\pi/32$', r'$\pi/16$', r'$\pi/8$', r'$\pi/4$', r'$\pi/2$'])
        colors = np.array(['#01295F', '#419D78', '#ECBA82', '#D1495B', '#98DFEA'])
        idx = 0
        for theta in thetaarr:
            ldarr = ld(htest, theta)
            plt.plot(htest, ldarr, label=r'$\theta = \,$' + labels[idx])
            idx += 1
        plt.ylabel(r'$\ell_d(h, \theta)\,\textrm{[km]}$')
        plt.xlabel(r'$h\,\textrm{[km]}$')
        plt.legend()
        plt.title(r'$\ell_d(h, \theta)$ for different values of $\theta$')
        plt.savefig('plots/ldexample.pdf', bbox='tight')

    plt.figure()
    htest = np.linspace(0.0, 60.0, 100)
    thetaarr = np.array([np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2])
    labels = np.array([r'$\pi/32$', r'$\pi/16$', r'$\pi/8$', r'$\pi/4$', r'$\pi/2$'])
    colors = np.array(['#01295F', '#419D78', '#ECBA82', '#D1495B', '#98DFEA'])
    idx = 0
    for theta in thetaarr:
        Yparr = np.empty(len(htest))
        for eval in range(0, len(htest)):
            Yparr[eval] = Yp(htest[eval], theta)
        plt.plot(htest, Yparr, label=r'$\theta = \,$' + labels[idx])
        idx += 1
    plt.ylabel(r'$Y_p(h, \theta)$')
    plt.xlabel(r'$h\,\textrm{[km]}$')
    plt.legend()
    axes = plt.axis()
    plt.axis([axes[0], axes[1], axes[2], 1.05])
    plt.title(r'$Y_p(h, \theta)$ for different values of $\theta$')
    plt.savefig('plots/Ypexample.pdf', bbox='tight')

    plt.figure()
    size = 100
    harr, thetaarr = np.meshgrid(np.linspace(0, 60, size), np.linspace(0, np.pi, size))
    Yparr = np.empty([size, size])
    for i in range(0, size):
        for j in range(0, size):
            Yparr[i][j] = Yp(harr[i][j], thetaarr[i][j])
    plt.contourf(harr, thetaarr, Yparr)
    plt.xlabel(r'$h \, \textrm{[km]}$')
    plt.ylabel(r'$\theta$')
    plt.colorbar(label=r'$Y_p(h, \theta)$')
    plt.title(r'Contour plot of $Y_p(h, \theta)$')
    plt.savefig('plots/Ypcontour.pdf', bbox='tight')
    '''
    # NA = 6.022*np.power(10.0, 23)
    # rarr = np.linspace(0.0, 6370.0 + 180.0, 10000)
    # rhoarr = rhoE(rarr)/np.power(10.0, 23)
    # plt.figure()
    # plt.plot(rarr/np.power(10.0, 3), rhoarr)
    # plt.xlabel(r'$r \, \textrm{[}10^{3}\,\textrm{km]}$')
    # plt.ylabel(r'$n_E \,\textrm{[}10^{23} \, \textrm{cm}^{-3}\textrm{]}$')
    # plt.grid(True)
    # plt.title(r'Number Density of the Earth')
    # plt.savefig('plots/nE.pdf')

    # RE = 6378.1
    # h = 100.0
    # thetaarr = np.array([np.pi, 3*np.pi/4, np.pi/2, np.pi/4])
    # labels = np.array([r'$\theta = \pi$', r'$\theta = 3\pi/4$', r'$\theta = \pi/2$', r'$\theta = \pi/4$'])
    # plt.figure()
    # plt.xlabel(r'$z\,\textrm{[}10^3\,\textrm{km]}$')
    # plt.ylabel(r'$r(z, h, \theta)\,\textrm{[}10^3\,\textrm{km]}$')
    # plt.title(r'$r(z, h, \theta)$ for $h = 100\,\textrm{km}$')
    # plt.text(1.0, 1.01*RE/10**3, r'$R_E$')
    # idx = 0
    # for theta in thetaarr:
    #     zarr = np.linspace(0, ld(h, theta), 1000)
    #     rarr = r(zarr, h, theta)
    #     plt.plot(zarr/10**3, rarr/10**3, label=labels[idx])
    #     idx += 1
    # plt.plot([0.0, ld(h, np.pi)/10**3], [RE/10**3, RE/10**3], c='k', ls='-.')
    # plt.fill_between([0.0, ld(h, np.pi)/10**3], [RE/10**3, RE/10**3], alpha=0.1, color='#56351E')
    # axes = plt.axis()
    # plt.fill_between([0.0, ld(h, np.pi)/10**3], [RE/10**3, RE/10**3], [axes[3], axes[3]], alpha=0.2, color='#4ECDC4')
    # plt.axis([axes[0], ld(h, np.pi)/10**3, axes[2], axes[3]])
    # plt.legend(loc='best', fontsize=8)
    # plt.savefig('plots/rexamples.pdf')
    '''
    plt.figure()
    htest = np.linspace(0.0, 180.0, 200)
    thetaarr = np.array([np.pi/64, np.pi/32, np.pi/24, np.pi/16])
    labels = np.array([r'$\pi/64$', r'$\pi/32$', r'$\pi/24$', r'$\pi/16$'])
    sigmachi = np.power(10.0, -32)
    idx = 0
    for theta in thetaarr:
        Ydarr = np.empty(len(htest))
        for eval in range(0, len(htest)):
            Ydarr[eval] = Yd(htest[eval], theta, sigmachi)
        plt.plot(htest, Ydarr, label=r'$\theta = \,$' + labels[idx])
        idx += 1
    plt.ylabel(r'$Y_d(h, \theta)$')
    plt.xlabel(r'$h\,\textrm{[km]}$')
    plt.legend()
    axes = plt.axis()
    plt.axis([axes[0], axes[1], axes[2], 1.05])
    plt.title(r'$Y_d(h, \theta)$ with $\sigma_\chi^{\textrm{SI}} = 10^{-32}\,\textrm{cm}^2$')
    plt.savefig('plots/Ydexample.pdf', bbox='tight')
    '''

    # plt.figure()
    # size = 50
    # sigmachi = np.power(10.0, -32)
    # harr, thetaarr = np.meshgrid(np.linspace(0, 60, size), np.linspace(0, np.pi, size))
    # Ydarr = np.empty([size, size])
    # for i in range(0, size):
    #     for j in range(0, size):
    #         Ydarr[i][j] = Yd(harr[i][j], thetaarr[i][j], sigmachi)
    # plt.contourf(harr, thetaarr, Ydarr)
    # plt.xlabel(r'$h \, \textrm{[km]}$')
    # plt.ylabel(r'$\theta$')
    # plt.colorbar(label=r'$Y_d(h, \theta)$')
    # plt.title(r'Contour plot of $Y_d(h, \theta)$')
    # plt.savefig('plots/Ydcontour.pdf', bbox='tight')
    #
    # plt.figure()
    # size = 1000
    # harr, thetaarr = np.meshgrid(np.linspace(0, 180, size), np.linspace(0, np.pi, size))
    # zarr = zstar(h=harr, theta=thetaarr)
    # plt.contourf(harr, thetaarr, zarr)
    # plt.xlabel(r'$h \, \textrm{[km]}$')
    # plt.ylabel(r'$\theta$')
    # plt.colorbar(label=r'$z_\star(h, \theta)\, \textrm{[km]}$')
    # plt.title(r'Contour plot of $z_\star(h, \theta)$')
    # plt.savefig('plots/zstarcontour.pdf', bbox='tight')

    # plt.figure()
    # htest = np.linspace(0.0, 180.0, 100)
    # thetaarr = np.array([np.pi/32, np.pi/16, np.pi/8])
    # labels = np.array([r'$\pi/32$', r'$\pi/16$', r'$\pi/8$'])
    # idx = 0
    # for theta in thetaarr:
    #     zarr = zstar(htest, theta)
    #     plt.plot(htest, zarr, label=r'$\theta = \,$' + labels[idx])
    #     idx += 1
    # plt.ylabel(r'$z_\star(h, \theta)\,\textrm{[km]}$')
    # plt.xlabel(r'$h\,\textrm{[km]}$')
    # plt.legend(fontsize=8)
    # plt.title(r'$z_\star(h, \theta)$ for different values of $\theta$')
    # plt.savefig('plots/zstarexample.pdf', bbox='tight')

    size = 100
    sigmaarray = np.logspace(-34.0, -28.0, size)
    # Gdetarr = np.empty(size)
    # for idx in range(0, size):
    #     Gdetarr[idx] = Gdet(sigmachi=sigmaarray[idx], mchi=0.001)
    # np.save('Gdetarr.npy', Gdetarr)
    Gdetarr = np.load('Gdetarr.npy')

    GdetFun = interp1d(sigmaarray, Gdetarr, kind='slinear')
    sigmatest = np.logspace(-34.0, -28.0, 1000)
    GdetFitarr = GdetFun(sigmatest)
    heffarr = GdetFitarr/(5.05*np.power(10.0, 24))

    fig, ax1 = plt.subplots()
    ax1.set_ylabel(r'$G_{\textrm{\small det}}(\sigma_\chi^{\textrm{\small SI}})\,\textrm{[cm}^{-2}\textrm{]}$')
    ax1.set_xlabel(r'$\sigma_\chi^{\textrm{\small SI}}\,\textrm{[cm}^2\textrm{]}$')


    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$h_{\textrm{\small eff}}(\sigma_\chi^{\textrm{\small SI}})\,\textrm{[km]}$')
    ax2.plot(sigmatest, heffarr)
    ax1.semilogx(sigmaarray, Gdetarr, '+', color='#D1495B', markersize=0.0, markeredgewidth=0.5)
    ax2.text(np.power(10.0, -30.6), 5.5, r'$m_\chi = 0.001\,\textrm{GeV}$', fontsize=10)
    ax2.text(np.power(10.0, -30.6), 5.2, r'$n_{\textrm{\small eff}} = 5.05 \times 10^{19}\,\textrm{cm}^{-3}$', fontsize=10)
    plt.title(r'The Effective Height')
    plt.savefig('plots/gdet.pdf')
