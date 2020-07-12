# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        BlackScholes
# Purpose:
#
# Author:      kklekota
#
# Created:     22/05/2014
# Copyright:   (c) kklekota 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from math import sqrt, fabs, log, pi

EPS = 1e-6

ImpliedGaussProxyNumCoeffs = [
   3.994961687345134e-1,
   2.100960795068497e+1,
   4.980340217855084e+1,
   5.988761102690991e+2,
   1.848489695437094e+3,
   6.106322407867059e+3,
   2.493415285349361e+4,
   1.266458051348246e+4
]

ImpliedGaussProxyDenomCoeffs = [
   1.000000000000000e+0,
   4.990534153589422e+1,
   3.093573936743112e+1,
   1.495105008310999e+3,
   1.323614537899738e+3,
   1.598919697679745e+4,
   2.392008891720782e+4,
   3.608817108375034e+3,
   -2.067719486400926e+2,
   1.174240599306013e+1
]

def _irsmBlackNormalProxyFuncImplied (dEta):
    dRes = 0.0
    dNum = 0.0
    dDenom = 0.0
    
    dPowEta = 1.0
    
    iNumN = 8
    iDenomN = 10
    
    for i in range(0, iNumN):
        dNum += ImpliedGaussProxyNumCoeffs[i] * dPowEta
        dDenom += ImpliedGaussProxyDenomCoeffs[i] * dPowEta
        dPowEta *= dEta
        
    for i in range(iNumN, iDenomN):
        dDenom += ImpliedGaussProxyDenomCoeffs[i] * dPowEta
        dPowEta *= dEta
        
    dRes = sqrt(dEta) * dNum / dDenom
    return dRes

def _irsmBlackNormalProxyFunc (db):
    dX = 0.0
    dAbsB = fabs(db)
    
    if dAbsB < 0.000001:
        dX = 1.0 - db * db / 3.0
    else:
        dX = 2 * db / log((1.0 + db) / (1.0 - db))
        
    return dX
        
def _irsmBlackNormalImpliedVolFG (dForward, dStrike, dPremium, w):
    dIntrinsic = w * (dForward - dStrike)
    
    dStraddle = 2.0 * dPremium - dIntrinsic
    dA = sqrt(2.0 * pi)
    dB = dIntrinsic / dStraddle
    dEta = _irsmBlackNormalProxyFunc(dB)
    
    dVolatilityFG = dA * dStraddle * _irsmBlackNormalProxyFuncImplied(dEta)
    
    return dVolatilityFG if dVolatilityFG > 0.0 else 0.02 

def BachelierPrice(F, K, v, w = 1.0):
    from scipy.stats import norm
    if abs(w) != 1.0: 
        raise ValueError('w should be 1.0 or -1.0.')
    if v <= 0: 
        raise ValueError('v should be positive.')
    x = (F - K) / v
    return v * (w * x * norm.cdf(w * x) + norm.pdf(x))

def BSPrice(F, K, v, w = 1.0):
    """Calculates Black&Scholes option price.

    Parameters
    ----------
    F : double
	    `F` is a forward value.
    K : double
        `K` is a strike.
    v : double
        `v` is a total volatility. Normally it's :math:`\sigma\sqrt{T}`.
    w : {-1.0, 1.0}, optional
        `w` is 1.0 for call option and -1.0 for put (the default is 1.0).
    
    Returns
    -------
    double
        Black Scholes price of the option.
    """
    from scipy.stats import norm
    if abs(w) != 1.0: 
        raise ValueError('w should be 1.0 or -1.0.')
    if v <= 0: 
        raise ValueError('v should be positive.')
    d1 = log(F / K) / v + v / 2
    d2 = d1 - v
    return F * w * norm.cdf(w * d1) - K * w * norm.cdf(w * d2)

def _corradoLnApp(price, F, K, w):
    d = w * (F - K) / 2
    return sqrt(2 * pi) / (F + K) \
        * (price - d + sqrt(max((price - d) ** 2 - 4 * d ** 2 / pi, 0)))

def _bharadiaLnApp(price, F, K, w):
    d = w * (F - K) / 2
    return sqrt(2 * pi) * (price - d) / (F - d)

def _getImpliedVol(price, F, K, w, priceF, getInvAtmF, getMinVolF, getMaxVolF):
    """Calculates implied vol.

    Parameters
    ----------
    price : double
	    Contract value.
    F : double
	    `F` is a forward value.
    K : double
        `K` is a strike.
    w : {-1.0, 1.0}, optional
        `w` is 1.0 for call option and -1.0 for put (the default is 1.0).

    Returns
    -------
    double
        Total volatility. Normally it's :math:`\sigma\sqrt{T}`.
    """
    from scipy.optimize import brentq

    if abs(w) != 1.0: 
        raise ValueError('w should be 1.0 or -1.0')

    if price <= w * (F - K):
        raise ValueError('Option value is smaller than intrinsic value')
    
    if abs(F - K) < 1e-6:
        return getInvAtmF(price, F, w)

    # If ITM we convert to OTM
    if w * (F - K) > 0.0:
        return _getImpliedVol(price - w * (F - K), F, K, -w, priceF, getInvAtmF, getMinVolF, getMaxVolF)

    f = lambda vol: priceF(F, K, vol, w) - price

    volMin = getMinVolF(price, F, K, w)
    while f(volMin) > 0 and volMin > 1e-6:
        volMin /= 2.0

    volMax = getMaxVolF(price, F, K, w)
    while f(volMax) < 0 and volMax < 5.:
        volMax *= 2.0

    if f(volMin) > 0:
        return volMin

    if f(volMax) < 0: 
        return volMax

    return brentq(f, volMin, volMax, xtol=1e-6, full_output=0, maxiter=100)


def ImpliedVol(price, F, K, w=1.0):
    from scipy.stats import norm

    if (w == 1.0 and price > F) or (w == -1 and price > K):
        raise ValueError('Premium is impossible')

    getInvAtmF = lambda pp, ff, ww: -2.0 * norm.ppf((1.0 - pp/ff) / 2.0) if pp/ff < 1.0 else 0.0
    getMinVolF = lambda pp, ff, kk, ww: 0.5 * _corradoLnApp(pp, ff, kk, ww)
    getMaxVolF = lambda pp, ff, kk, ww: 2.0 * _bharadiaLnApp(pp, ff, kk, ww)
    return _getImpliedVol(price, F, K, w, BSPrice, getInvAtmF, getMinVolF, getMaxVolF)


def ImpliedNormalVol(price, F, K, w = 1.0):
    
    if price > BachelierPrice(F, K, max(abs(F * 10.0), 1.0), w):
        raise ValueError('Premium is too high')

    getInvAtmF = lambda pp, ff, ww: pp * sqrt(2.0 * pi)
    getMinVolF = lambda pp, ff, kk, ww: max(0.5 * _irsmBlackNormalImpliedVolFG(ff, kk, pp, ww), 1e-5)
    getMaxVolF = lambda pp, ff, kk, ww: 2.0 * _irsmBlackNormalImpliedVolFG(ff, kk, pp, ww)
    return _getImpliedVol(price, F, K, w, BachelierPrice, getInvAtmF, getMinVolF, getMaxVolF)


# f_ref = 0.0040449536048
# T = 2.0
# shift = 0.005
# vol_sln_ref = 0.3929925987888
# strike = 0.4046566778538 * 1e-2
# annuity = 1.9975146942704
#
# price_ref = annuity * BSPrice(f_ref + shift, strike + shift, vol_sln_ref * sqrt(T))
#
# f_bump = 0.005049684929
# annuity_bump = 1.9905132432836
# sigma_norm = vol_sln_ref * (f_ref + shift) * (1.0 - vol_sln_ref ** 2.0 * T / 24.)
# vol_sln_bump = sigma_norm / (f_bump + shift) * (1.0 + (sigma_norm / (f_bump + shift)) ** 2.0 * T / 24.0)
# premium_new = annuity_bump * BSPrice(f_bump + shift, strike + shift, vol_sln_bump * sqrt(T))
#
# dv01 = (premium_new - price_ref) / 10.0
# 1000000.000000 * dv01
# 1000000.000000 * price_ref
# 1000000.000000 * premium_new

