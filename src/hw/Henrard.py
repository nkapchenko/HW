import numpy as np
from numpy import exp, sqrt, array
from functools import partial
from scipy import optimize
from scipy.stats import norm
import scipy.integrate as integrate
from collections import namedtuple
from copy import deepcopy
from fox_toolbox.utils import rates
from hw import hw_helper


"""This module price swaption under Hull White model using Hernard method

Usage example:

from hw import Henrard
henr_price, debug = Henrard.hw_swo(swo, ref_mr, sigma_hw_henr, dsc_curve, estim_curve)

swo            : rates.Swaption
ref_mr         : float
sigma_hw_henr  : rates.Curve
dsc_curve      : rates.RateCurve
estim_curve    : rates.RateCurve
"""


def get_coef(dcfs, strike, flt_adjs):
    """ Coefficients for Put swaption. Henrard  """
    c0 = -1.
    c = [c0] + list(map(lambda dcf, fadj: dcf * (strike - fadj), dcfs, flt_adjs))
    c[-1] += 1
    return np.array(c)


def get_P_i(dsc_curve, hw_dates):
    return [dsc_curve.get_dsc(t) for t in hw_dates]


def integral_alpha(t1, t2, a):
    "integral_t1^t2 exp(2as) ds"
    return (exp(2 * a * t2) - exp(2 * a * t1))/(a * 2)

def get_alpha_i(a: float, expiry:float, hw_dates: np.ndarray, sigma: rates.Curve):
    
    integral_func = lambda t1, t2 : integral_alpha(t1, t2, a)
    total_var = hw_helper.hw_integrate(sigma=sigma, func=integral_func, T1=0, T2 = expiry, a=a)
    alpha_i = [sqrt(total_var) * (exp(-a * expiry) - exp(-a * t)) / a for t in hw_dates]
    return array(alpha_i), total_var


def swap_value(coef, P_i, alpha_i, x):
    """ Swap function for finding x_star """
    pi_exp_alpha = P_i * exp(- alpha_i * x - 0.5 * alpha_i ** 2)
    return coef.dot(pi_exp_alpha)


def get_x_star(coef, P_i, alpha_i):
    assert len(hw_helper.sign_changes(coef)) >= 1, 'all Henrard coefficients are of the same sign'
    func = partial(swap_value, coef, P_i, alpha_i)
    return optimize.newton(func, x0=0., tol=1.e-06, maxiter=50)


def hw_swo_analytic(coef, P_i, alpha_i, x_star, IsCall):
    sign = -1 if IsCall else 1
    if IsCall: coef = np.negative(coef)
    values = P_i * norm.cdf(sign*(x_star + alpha_i))
    return coef.dot(values)


def hw_swo_numeric(coef, P_i, alpha_i, IsCall):

    if IsCall: coef = np.negative(coef)
    return integrate.quad(lambda x: swo_payoff(coef, P_i, alpha_i, x) * norm.pdf(x), -6, 6)[0]


def swo_payoff(coef, P_i, alpha_i, x):
    """Call/Put is hidden in coef"""
    swap = swap_value(coef, P_i, alpha_i, x)
    return swap if swap > 0 else 0


def numerical_correction(coef, P_i, alpha_i, sgn_changes, IsCall):
    """compute correction as difference between analytic HW with degenerated coefficients and numerical integration"""
    
    hw_num_debug = namedtuple('hw_num_debug', 'degen_coef degen_x_star degen_anal_price degen_num_price correction')

    degen_coef = deepcopy(coef)
    degen_coef[sgn_changes[1] + 1: sgn_changes[-1] + 1].fill(0.)
    degen_x_star = get_x_star(degen_coef, P_i, alpha_i)

    degen_anal_price = hw_swo_analytic(degen_coef, P_i, alpha_i, degen_x_star, IsCall)
    degen_num_price = hw_swo_numeric(degen_coef, P_i, alpha_i, IsCall)
    correction = degen_anal_price - degen_num_price

    return correction, hw_num_debug(degen_coef, degen_x_star, degen_anal_price, degen_num_price, correction)


def hw_swo(swo, a, sigma, dsc_curve, estim_curve):
    assert isinstance(sigma, rates.Curve), 'sigma should be rates.Curve object'

    IsCall = False if swo.pay_rec == 'Receiver' else True
    flt_adjs = swo.get_flt_adjustments(dsc_curve, estim_curve)

    hw_dates = np.insert(swo.payment_dates, 0, swo.start_date)
    coef = get_coef(swo.day_count_fractions, swo.strike, flt_adjs)
    P_i = get_P_i(dsc_curve, hw_dates)
    alpha_i, total_var = get_alpha_i(a, swo.expiry, hw_dates, sigma)
    
    sgn_changes = hw_helper.sign_changes(coef)
    change_once = len(sgn_changes) == 1
    

    if change_once:
        x_star = get_x_star(coef, P_i, alpha_i)
        debug_dict = {'Expiry': swo.expiry,
                 'hw_dates': hw_dates,
                 'flt_adjs' : [np.nan] + list(flt_adjs),
                 'coef': coef,
                 'P_i': P_i,
                 'alpha_i': alpha_i,
                 'x_star': x_star,             
                 'IsAnalytic': change_once,
                 'total_var' : total_var,
                 }
        return hw_swo_analytic(coef, P_i, alpha_i, x_star, IsCall), debug_dict
    else:
        correction, num_debug = numerical_correction(coef, P_i, alpha_i, sgn_changes, IsCall)
        debug_dict = {'Expiry': swo.expiry,
                 'hw_dates': hw_dates,
                 'flt_adjs' :[np.nan] + list(flt_adjs),
                 'coef': coef,
                 'P_i' : P_i,
                 'alpha_i': alpha_i,
                 'x_star' : np.nan,               
                 'IsAnalytic': change_once,
                 'degen_coef': num_debug.degen_coef,
                 'degen_x_star': num_debug.degen_x_star,
                 'total_var' : total_var,
                 'correction': num_debug.correction,
                 }
        return hw_swo_numeric(coef, P_i, alpha_i, IsCall) + correction, debug_dict
    
    

"""
Ref:
Notes on Henrard swaption evaluation. Murex (c) 
"""

def old_get_alpha_i(a: float, expiry:float, hw_dates: np.ndarray, sigma):
    assert hw_dates[0] > expiry, 'hw_dates should start after swo expiry'
    assert  isinstance(sigma, (float, rates.Curve)), f'sigma: float or rates.Curve, not {type(sigma)}'


    if isinstance(sigma, float):
        total_var = np.sqrt(0.5 * (sigma ** 2) * (np.exp(2 * a * expiry) - 1) / a)
        res = [total_var * (np.exp(-a * expiry) - np.exp(-a * t)) / a for t in hw_dates]
        return np.array(res)
    elif isinstance(sigma, rates.Curve):
        total_var = 0.
        previous_expiries = [t_exp for t_exp in sigma.buckets if t_exp <= expiry]
        previous_sigmas = sigma.values[:len(previous_expiries)]

        if previous_expiries[-1] < expiry:
            previous_sigmas.append(sigma.values[len(previous_expiries)])
            previous_expiries.append(expiry)

        for i in range(len(previous_expiries) - 1):
            total_var += 0.5 * (previous_sigmas[i+1] ** 2) * (np.exp(2 * a * previous_expiries[i+1]) - np.exp(2 * a * previous_expiries[i])) / a

        alpha_i = [sqrt(total_var) * (np.exp(-a * expiry) - np.exp(-a * t)) / a for t in hw_dates]
        return np.array(alpha_i), total_var