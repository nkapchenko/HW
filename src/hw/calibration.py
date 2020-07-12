from numpy import exp, sqrt, set_printoptions, array
import pandas as pd
from math import pi
from scipy.stats import norm
from scipy import optimize
from functools import partial
from fox_toolbox.utils import volatility as vols
from fox_toolbox.utils import rates
from hw.Jamshidian import hw_swo as hw_swo_jamsh
from hw.Henrard import hw_swo as hw_swo_henr
from collections import namedtuple
import warnings

set_printoptions(precision=2)

"""This module price swaption under Hull White model using Jamshidian or Henrard method.

Usage example:

from hw import calibration as hw_calib

sigma_hw, debug = hw_calib.calibrate_sigma_hw(cal_basket, mr, dsc_curve, estim_curve, IsJamsh=False)

cal_basket     : list of rates.Swaption
mr             : float
dsc_curve      : rates.RateCurve
estim_curve    : rates.RateCurve
IsJamsh        : bool (True - Jamshidian, False - Henrard)
"""

EPS = 1e-14


def _C(swo, a, curve, estim_curve):

    flt_adjs = swo.get_flt_adjustments(curve, estim_curve)
    fwd = swo.get_swap_rate(curve, None, flt_adjs)
    annuity = swo.get_annuity(curve)

    def eap(t):
        return exp(-a * t) / a * curve.get_dsc(t) / annuity


    return eap(swo.start_date) - eap(swo.maturity) + sum(map(lambda dcf, t, f: dcf * (fwd - f) * eap(t),
                                                                 swo.day_count_fractions, swo.payment_dates, flt_adjs))



def calibrate_sigma_hw(cal_basket, a, curve, estim_curve, IsJamsh=True):
    
    """
    var_hw IS NOT hw total variance, it is the function V(T) (see macs page 12.)
    Calibration: we DON'T match mkt_var, we match swo price.
    mkt_var is used to estimate first guess.
    """
    
    previous_expiry = 0.
    var_hw = 0.
    sigma1d = None
    calib_debug = {key: [] for key in 'expiries v0Schrager sigma_hw mkt_var tvar_schrager var_hw_V(T) target_price model_price'.split()}

    print(f'Starting calibration on {len(cal_basket)} swaptions with vol type: {cal_basket[0].vol.type}')
    for swo in cal_basket:

        flt_adjs = swo.get_flt_adjustments(curve, estim_curve)
        fwd = swo.get_swap_rate(curve, None, flt_adjs)

        w = -1 if swo.pay_rec == 'Receiver' else 1
        calib_annuity = swo.get_annuity(curve)
        
        if swo.vol.type == 'N':
            market_swo_price = calib_annuity * vols.BachelierPrice(fwd, swo.strike, swo.vol.value * sqrt(swo.expiry), w)
        else:
            market_swo_price = calib_annuity * vols.BSPrice(fwd, swo.strike, swo.vol.value*sqrt(swo.expiry), w)
        
        # debug
        mkt_var = swo.vol.value ** 2 * swo.expiry     
        c2 = _C(swo, a, curve, estim_curve)**2
        factor = (exp(2 * a * swo.expiry) - exp(2 * a * previous_expiry)) / (2 * a)
        
        vol0_guess = sigma_schrager(swo, previous_expiry, a, curve, estim_curve, var_hw)
        
        if vol0_guess is False:
            vol0 = vol0
        else:
            vol0 = vol0_guess
            tvar_schrager = c2 * (var_hw + vol0**2 * factor)
            assert abs(mkt_var - tvar_schrager) < EPS, f'vol0 should match mkt var by default.'
         

        hw_swo = hw_swo_jamsh if IsJamsh else hw_swo_henr
        _hw_swo = partial(hw_swo, swo, a, dsc_curve=curve, estim_curve=estim_curve)

        def price_diff(sigma, sigma1d):
            sigma1d = sigma1d_update(sigma1d, swo.expiry, sigma)
            hw_swo_price, _debug = _hw_swo(sigma1d)
            return hw_swo_price - market_swo_price

        optimum_sigma = optimize.newton(price_diff, x0=vol0, args=(sigma1d,), tol=1.e-09, maxiter=80)
        sigma1d = sigma1d_update(sigma1d, swo.expiry, optimum_sigma)
        
        var_hw += optimum_sigma ** 2 * factor
        previous_expiry = swo.expiry
        
        
        
        model_price, _ = _hw_swo(sigma1d)
        for key, val in zip(calib_debug.keys(), [swo.expiry, vol0, optimum_sigma, mkt_var, tvar_schrager, var_hw, market_swo_price, model_price]):
            calib_debug[key].append(val)
            
    # extrapolate left and right values:
    sigma1d = sigma1d_update(sigma1d, swo.expiry + 30., optimum_sigma) # extra on the right
    sigma1d = rates.Curve(sigma1d.buckets, [sigma1d.values[1]] + list(sigma1d.values)[1:], sigma1d.interpolation_mode, sigma1d.label + ('Jamshidian' if IsJamsh else 'Henrard'))
    
    print('Calibration SUCCESSFUL') if max(array(calib_debug['model_price']) - array(calib_debug['target_price'])) < EPS else print('Cabration PROBLEM !')
    return CalibrationHW(sigma1d, calib_debug)


def sigma1d_update(sigma1d, expiry, sigma):
    if sigma1d is None:
        return  rates.Curve([0., expiry], [None, sigma], 'PieceWise', 'HW model sigma ')
    if expiry not in sigma1d.buckets:
        buckets = list(sigma1d.buckets) + [expiry]
        sigmas = list(sigma1d.values) + [sigma]
    else:
        buckets = sigma1d.buckets
        sigmas = list(sigma1d.values)[:-1] + [sigma]
        
    return rates.Curve(buckets, sigmas, sigma1d.interpolation_mode, sigma1d.label)




def sigma_schrager(swo, previous_expiry, a, curve, estim_curve, previous_hw_var):
    """Initial guess for first period
    previous_hw_var is V(T_previous_expiry) WITHOUT C(a)**2 !
    """

    mkt_variance = swo.vol.value ** 2 * swo.expiry

    factor = (exp(2 * a * swo.expiry) - exp(2 * a * previous_expiry)) / (2 * a)
    assert factor > 0, f'HW: either negative meanRev {a} or swo.expiry {swo.expiry} < previous_expiry {previous_expiry}'
    c = _C(swo, a, curve, estim_curve)
    
    '>>> hw_variance = c**2 V(T_exp) = c**2 * previous_hw_var + c**2 * sigma_hw_T **2 * factor = mkt_variance <<<'

    if (mkt_variance - c**2 *  previous_hw_var) < 0:
        warnings.warn(f'Schrager: Lack of vol to match market total variance at T = {swo.expiry} \n market_variance {mkt_variance}\n Schrager previous variance {c**2 * previous_hw_var}. ')
        return False
    
    
    return sqrt( (mkt_variance - c**2 * previous_hw_var) / (factor * c ** 2) )

class CalibrationHW():
    
    def __init__(self, sigma, calib_debug):
        self.sigma = sigma
        self.debug = calib_debug
        self.data = pd.DataFrame(calib_debug)
        
    def plot(self, irsmout=None):
        from hw.vizual import calib_plot
        calib_plot(self.sigma, self.debug, irsmout)





        






