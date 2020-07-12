from functools import partial

import numpy as np
from scipy import optimize
from scipy.stats import norm
from scipy.integrate import quad

from script.fox_toolbox.utils import rates


def get_hw_dates(cal_instr):
    if not isinstance(cal_instr, rates.Swap):
        raise TypeError()
    hw_dates = np.insert(cal_instr.payment_dates, 0, cal_instr.start_date)
    return hw_dates


def get_hw_weights(cal_instr, strike, main_curve, fwd_curve=None, is_payer=True):
    if not isinstance(cal_instr, rates.Swap):
        raise TypeError()
    if not isinstance(main_curve, rates.Curve):
        raise TypeError()
    if fwd_curve is not None and not isinstance(main_curve, rates.Curve):
        raise TypeError()

    if fwd_curve is not None:
        flt_adjs = cal_instr.get_flt_adjustments(main_curve, fwd_curve)
    else:
        flt_adjs = np.zeros_like(cal_instr.payment_dates)

    strike = float(strike)
    hw_coeffs = np.insert(cal_instr.day_count_fractions * (strike - flt_adjs), 0, -1.0)
    hw_coeffs[-1] += 1
    return hw_coeffs if is_payer else -hw_coeffs


def get_hw_alphas(sigma, mr, expiry, hw_dates):
    if not isinstance(sigma, float) or not isinstance(mr, float) \
            or not isinstance(expiry, float) or not isinstance(hw_dates, np.ndarray):
        raise TypeError()
    alphas = np.sqrt(np.square(sigma / mr * (np.exp(-mr * expiry) - np.exp(-mr * hw_dates)))
                     * (np.exp(2.0 * mr * expiry) - 1.0) / (2.0 * mr))
    return alphas


def get_swap_price(hw_coeffs, hw_discounts, hw_alphas, x):
    swap_price = np.dot(hw_coeffs * hw_discounts, np.exp(-hw_alphas * x - 0.5 * np.square(hw_alphas)))
    return swap_price


def get_x_prime(hw_coeffs, hw_discounts, hw_alphas):
    # TODO: check if hw_coeffs change sign; otherwise need to use integration
    # TODO: initial guess for x0
    x0 = 0.0
    pricer = partial(get_swap_price, hw_coeffs, hw_discounts, hw_alphas)
    opt_res = optimize.root(pricer, x0=x0)
    x_prime = opt_res.x[0]
    return x_prime


def get_hw_premium(hw_coeffs, hw_discounts, hw_alphas, x_prime):
    premium = np.dot(hw_coeffs * hw_discounts, norm.cdf(x_prime + hw_alphas))
    return premium


def get_hw_premium_integration(hw_coeffs, hw_discounts, hw_alphas):
    def int_func(x):
        return max(get_swap_price(hw_coeffs, hw_discounts, hw_alphas, x), 0.0) * norm.pdf(x)
    premium, err = quad(int_func, -np.inf, np.inf)
    return premium
