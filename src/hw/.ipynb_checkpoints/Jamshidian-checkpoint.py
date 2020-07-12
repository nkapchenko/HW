import numpy as np
from numpy import exp, sqrt
from functools import partial
from scipy import optimize
from scipy.stats import norm
import scipy.integrate as integrate
from fox_toolbox.utils import rates
from hw.hw_helper import _B, _V, _A, sign_changes
from hw import hw_helper
from hw.hw_helper import get_var_x

"""This module price swaption under Hull White model using Jamshidian method.

Usage example:

from hw import Jamshidian as jamsh

jamsh_price, debug = jamsh.hw_swo(swo, ref_mr, sigma_hw_jamsh, dsc_curve, estim_curve)
swo            : rates.Swaption
ref_mr         : float
sigma_hw_jamsh : rates.Curve
dsc_curve      : rates.RateCurve
estim_curve    : rates.RateCurve
"""



def get_coef(swo, a, sigma, dsc_curve, estim_curve):
    """ Coefficients for Put swaption from calibration basket. Jamishidian  """
    flt_adjs = swo.get_flt_adjustments(dsc_curve, estim_curve)

    c0 = -_A(swo.expiry, swo.start_date, a, sigma, dsc_curve)
    c = list(map(lambda dcf, pdate, fadj: dcf * (swo.strike - fadj) * _A(swo.expiry, pdate, a, sigma, dsc_curve),
                 swo.day_count_fractions, swo.payment_dates, flt_adjs))
    c[-1] += _A(swo.expiry, swo.maturity, a, sigma, dsc_curve)
    c.insert(0, c0)
    return np.array(c)



def get_b_i(swo, a):
    """ array of B_i for by each payment date """
    b0 = _B(swo.expiry, swo.start_date, a)
    b = list(map(lambda pdate: _B(swo.expiry, pdate, a), swo.payment_dates))
    b.insert(0, b0)
    return np.array(b)


def swap_value(coef, b_i, varx, x):
    """ Swap function for finding x_star """
    exp_b_var = exp(- b_i * sqrt(varx) * x)
    return coef.dot(exp_b_var)


def get_x_star(coef, b_i, varx):
    x0 = .0
    func = partial(swap_value, coef, b_i, varx)
    # optimum = optimize.newton(func, x0=x0)
    optimum = optimize.bisect(func, -6, 6)
    return optimum


def hw_swo_analytic(coef, b_i, varx, x_star, IsCall):
    """ analytic """

    sign = -1 if IsCall else 1
    if IsCall: coef = np.negative(coef)

    val_arr = exp(0.5 * b_i ** 2 * varx) * norm.cdf(sign*(x_star + b_i * sqrt(varx)))

    return coef.dot(val_arr)


def hw_swo_numeric(coef, b_i, varx, IsCall):

    if IsCall: coef = np.negative(coef)

    swaption_numeric = integrate.quad(lambda x: swo_payoff(coef, b_i, varx, x) * norm.pdf(x), -10, 10)[0]

    degen_swo_analytic, degen_swo_numeric = 0, 0
    control_variable = degen_swo_analytic - degen_swo_numeric

    return swaption_numeric + control_variable


def swo_payoff(coef, b_i, varx, x):
    """Call/Put is hidden in coef"""
    swap = swap_value(coef, b_i, varx, x)
    return swap if swap > 0 else 0


def hw_swo(swo, a, sigma, dsc_curve, estim_curve):
    """ Main Hull White swaption function """
    IsCall = False if swo.pay_rec == 'Receiver' else True

    coef = get_coef(swo, a, sigma, dsc_curve, estim_curve)
    b_i = get_b_i(swo, a)
    varx = get_var_x(swo.expiry, a, sigma)
    
   
    sgn_changes = sign_changes(coef)
    change_once = len(sgn_changes) == 1
    
    
    if change_once:
        x_star = get_x_star(coef, b_i, varx)
        debug_dict = {}
        return hw_swo_analytic(coef, b_i, varx, x_star, IsCall), debug_dict
    else:
        debug_dict = {}
        return hw_swo_numeric(coef, b_i, varx, IsCall), debug_dict




