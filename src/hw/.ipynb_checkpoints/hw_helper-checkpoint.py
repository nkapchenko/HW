import numpy as np
from numpy import exp, sqrt, array
from fox_toolbox.utils import rates
import warnings


    
   
    

def sign_changes(array):
    """return number of times the sign is changed in array"""
    return np.where(np.diff(np.sign(array)))[0]

EPS = 1e-5

def cast_to_Curve(sigma):
    if isinstance(sigma, rates.Curve):
        return sigma
    elif isinstance(sigma, float):
        buckets = np.array([0, 30.])
        values = np.array([sigma] * 2)
        return rates.Curve(buckets, values, 'PieceWise', 'HW volatility')
    else:
        TypeError(f'sigma should be of type float or rates.Curve but not {type(sigma)}')
    


def _B(t, T, a):
    if abs(a) < EPS:
        return T - t
    return (1 - exp(-a * (T - t))) / a


def _V(t, T, u, a, sigma):  
    
    def integral_v(t1, t2, u, a):
        p1 = (t2 - t1)
        p2 = - (2 / a) * exp(-a * u) * (exp(a * t2) - exp(a * t1))
        p3 = exp(-2 * a * u) * (exp(2 * a *t2) - exp(2 * a *t1)) / (2 * a)
        return (p1 + p2 + p3) / (a**2)

    sigma = cast_to_Curve(sigma)
    integral_func = lambda t1, t2: integral_v(t1, t2, u, a)
    return hw_integrate(sigma=sigma, func=integral_func, T1=t, T2=T, a=a)



def _A(t, T, a, sigma, dsc_curve):
    sigma = cast_to_Curve(sigma)
    fwd_dsc = dsc_curve.get_fwd_dsc(t, T)
    return fwd_dsc * exp(0.5*(_V(0, t, t, a, sigma) - _V(0, t, T, a, sigma)))


def get_var_x(T, a, sigma, s=0):
    """T ~ expiry
    a ~ mean reversion"""
    
    def integral_var_x(t1, t2, T, a):
        return (exp(-2 * a * (T - t2)) - exp(-2 * a * (T - t1)))/(2 * a) 
    
    sigma = cast_to_Curve(sigma)  
    integrate_func = lambda t1, t2: integral_var_x(t1, t2, T=T, a=a)
    total_var = hw_integrate(sigma=sigma , func = integrate_func, T1=s, T2=T, a=a)
    return total_var


def get_drift_T(s, t, U, a, sigma):
    """
    s - is a lower bound in integral (is a Filtration time Fs)
    t - is a upper bound in integral (is a time point where x(t) is evaluated)
    U - is a measure maturity
    
    p.s. This function is reconciled with QuantLib.HullWhiteForwardProcess.M_T
    """   

    sigma = cast_to_Curve(sigma)   
    integral_func = lambda t1, t2 :integral_mT(t1, t2, t=t, U=U, a=a)
    return hw_integrate(sigma, integral_func, T1=s, T2=t, a=a)

def integral_mT(t1, t2, t, U, a):
    """
    t1, t2 are integral bounds
    t is a time point where x(t) is evaluated
    """
    p1 = exp(-a * (t - t2)) - exp(-a * (t - t1))
    p2 = exp(-a * (U + t - 2 * t2)) - exp(-a * (U + t - 2 * t1))
    return p1/a**2 - p2 / (2 * a**2)


def hw_integrate(sigma, func, T1, T2, a):
    """
    sigma - rates.Curve (HW piecewise vols)
    func - any deterministic function func(t1, t2)
            t1 - lower integration 
            t2 - upper integration
    T1 - lower bound (normally zero)
    T2 - upper bound (normally expiry)
    a - mean rev
    """
    if not np.isclose(func(0,1) + func(1,2), func (0,2), 1e-14):
        warnings.warn("Nikita: integrated function depends on integration bounds. Integrals are not additive. You can no longer reconcile values with flat sigma ")
    
    assert T2<=sigma.buckets[-1], f'HW sigma doesnt cover T={T2}. The last sigma date is {sigma.buckets[-1]}'
    
    ts = array( [T1] + [t_ for t_ in sigma.buckets[  :-1 ] if T1 < t_ < T2]        )
    Ts = array(        [T_ for T_ in sigma.buckets[ 1:   ] if T1 < T_ < T2] + [T2] )
    sigmas2 = sigma.values[1:len(ts)+1]**2
    
    
#     return sigmas2.dot(func(ts, Ts))

    return (sigma(Ts)**2).dot(func(ts, Ts))



def get_var_p(t,T,a,sigma, dsc_curve):
    
    a2 = _A(t,T,a,sigma, dsc_curve)**2
    bsigma2 = _B(t,T,a)**2 * get_var_x(t,a,sigma)
    
    return a2 * (exp(bsigma2) - 1) * exp(bsigma2)


def drift_const(s, t, U, mr):
    "Brigo solution p76 for constant sigma"
    p1 = 1.0 - np.exp(-mr * (t - s))
    p2 = np.exp(-mr * (U - t)) - np.exp(-mr * (U + t- 2 * s))
    return (1/mr) ** 2 * (p1 - p2 / 2.0)
