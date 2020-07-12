import numpy as np
from scipy.stats import norm
from scipy import integrate
from scipy import optimize
from fox_toolbox.utils.rates import Volatility


ABS_BND = 5.5134666259538827


def get_premium(fwd1: float, vol1: Volatility,
                fwd2: float, vol2: Volatility,
                corr: float, expiry: float, strike: float):

    if vol1.type != 'SLN' or vol2.type != 'SLN':
        raise NotImplemented()

    if corr == 1.0:
        return np.nan

    f1 = float(fwd1) + float(vol1.shift_size)
    f2 = float(fwd2) + float(vol2.shift_size)
    v1 = float(vol1.value) * np.sqrt(expiry)
    v2 = float(vol2.value) * np.sqrt(expiry)

    def func(u):
        return (f1 * np.exp(v1 * corr * u - (corr * v1) ** 2 / 2.0) * norm.cdf(
                        1.0 / (np.sqrt(1 - corr * corr) * v1) * np.log((
                            f1 * np.exp(v1 * corr * u + (0.5 - corr * corr) * v1 * v1)) / (
                            f2 * np.exp(v2 * u - 0.5 * v2 * v2) + strike))
                        )
                - (f2 * np.exp(v2 * u - 0.5 * v2 * v2) + strike) * norm.cdf(
                        1.0 / (np.sqrt(1 - corr * corr) * v1) * np.log((
                            f1 * np.exp(v1 * corr * u - 0.5 * v1 * v1)) / (
                            f2 * np.exp(v2 * u - 0.5 * v2 * v2) + strike))
                        )
                ) * norm.pdf(u)

    # res = quad(func, -ABS_BND, ABS_BND, epsabs=1e-7)
    res = integrate.quadrature(func, -ABS_BND, ABS_BND)
    return res[0]


def get_impl_corr(non_dsc_tgt: float,
                  fwd1: float, vol1: Volatility,
                  fwd2: float, vol2: Volatility,
                  expiry: float, strike: float):

    def func(corr):
        res = get_premium(fwd1, vol1, fwd2, vol2, corr, expiry, strike)
        return np.square(res - non_dsc_tgt)

    return optimize.root(func, [0.8]).x[0]

    # return brentq(func, 0.4, 0.999, xtol=1e-4, full_output=0, maxiter=100)
