import numpy as np
from numpy import exp, sqrt
from functools import partial
from scipy import optimize
from scipy.stats import norm
import scipy.integrate as integrate
from fox_toolbox.utils import rates

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

class Jamshidian():
    
    def __init__(self, mr, sigma, dsc_curve, estim_curve):
        
        assert  isinstance(sigma, (float, rates.Curve)), f'sigma: float or rates.Curve, not {type(sigma)}'
        
        self.mr = mr
        self.sigma = sigma
        self.dsc_curve = dsc_curve
        self.estim_curve = estim_curve
    
    @staticmethod
    def sign_changes(array):
        """return number of times the sign is changed in array"""
        return np.where(np.diff(np.sign(array)))[0]
    
    @staticmethod
    def _B(t, T, a):
        return (1 - exp(-a * (T - t))) / a
    
    @staticmethod
    def _v(t, T, u, a):
        p1 = (T - t)
        p2 = - (2 / a) * exp(-a * u) * (exp(a * T) - exp(a * t))
        p3 = exp(-2 * a *u) * (exp(2 * a *T) - exp(2 * a *t)) / (2 * a)
        return (p1 + p2 + p3) / (a**2)
    
    @staticmethod
    def _V(t, T, u, a, sigma):
        if isinstance(sigma, float):
            return sigma**2 * _v(t, T, u, a)
        elif isinstance(sigma, rates.Curve):
            total_var = 0.
            expiry = T
            previous_expiries = [t_exp for t_exp in sigma.buckets if t_exp <= expiry]
            previous_sigmas = list(sigma.values[:len(previous_expiries)])

            if previous_expiries[-1] < expiry:
                previous_sigmas.append(sigma.values[len(previous_expiries)])
                previous_expiries.append(expiry)

            for i in range(len(previous_expiries) - 1):
                total_var += (previous_sigmas[i+1] ** 2) * _v(t, previous_expiries[i+1], u, a)

            return total_var


    @staticmethod
    def _A(t, T, a, sigma, dsc_curve):
        assert  isinstance(sigma, (float, rates.Curve)), f'sigma: float or rates.Curve, not {type(sigma)}'
        fwd_dsc = dsc_curve.get_fwd_dsc(t, T)
        return fwd_dsc * exp(0.5*(_V(0, t, t, a, sigma) - _V(0, t, T, a, sigma)))

    def get_coef(self, swo):
        """ Coefficients for Put swaption from calibration basket. Jamishidian  """
        flt_adjs = swo.get_flt_adjustments(self.dsc_curve, self.estim_curve)

        c0 = -_A(swo.expiry, swo.start_date, self.mr, self.sigma, self.dsc_curve)
        c = list(map(lambda dcf, pdate, fadj: dcf * (swo.strike - fadj) * _A(swo.expiry, pdate, self.mr, self.sigma, self.dsc_curve),
                     swo.day_count_fractions, swo.payment_dates, flt_adjs))
        c[-1] += _A(swo.expiry, swo.maturity, self.mr, self.sigma, self.dsc_curve)
        c.insert(0, c0)
        return np.array(c)


    def get_var_x(self, expiry):

        if isinstance(sigma, float):
            return 1 / (2 * a) * (1 - exp(-2 * a * expiry)) * sigma ** 2 

        elif isinstance(sigma, rates.Curve):
            total_var = 0.
            previous_expiries = [t_exp for t_exp in self.sigma.buckets if t_exp <= expiry]
            previous_sigmas = list(self.sigma.values[:len(previous_expiries)])

            if previous_expiries[-1] < expiry:
                previous_sigmas.append(self.sigma.values[len(previous_expiries)])
                previous_expiries.append(expiry)

            for i in range(len(previous_expiries) - 1):
                total_var += 1 / (2 * self.mr) * (previous_sigmas[i+1] ** 2) * (exp(-2 * self.mr * (expiry - previous_expiries[i+1])) - exp(-2 * self.mr * (expiry - previous_expiries[i])))

            return total_var


    def get_b_i(self, swo):
        """ array of B_i for by each payment date """
        b0 = _B(swo.expiry, swo.start_date, self.mr)
        b = list(map(lambda pdate: _B(swo.expiry, pdate, self.mr), swo.payment_dates))
        b.insert(0, b0)
        return np.array(b)

    @staticmethod
    def swap_value(coef, b_i, varx, x):
        """ Swap function for finding x_star """
        exp_b_var = exp(- b_i * sqrt(varx) * x)
        return coef.dot(exp_b_var)

    @staticmethod
    def get_x_star(coef, b_i, varx):
        x0 = .0
        func = partial(swap_value, coef, b_i, varx)
        # optimum = optimize.newton(func, x0=x0)
        optimum = optimize.bisect(func, -6, 6)
        return optimum
    
###TODO: continue adopting


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




