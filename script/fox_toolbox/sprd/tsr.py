import numpy as np
from scipy import optimize

from fox_toolbox.utils import rates

EPS = 1e-5


def get_b(mr: float, start_date, end_date):
    if abs(mr) < EPS:
        return end_date - start_date
    return (1.0 - np.exp(-mr * (end_date - start_date))) / mr


def get_linear_coeffs(cal_instr, main_curve, mr: float, pmnt_date: float, fwd_curve=None):
    if not isinstance(cal_instr, rates.Swap):
        raise TypeError()
    if not isinstance(main_curve, rates.Curve):
        raise TypeError()
    if fwd_curve is not None and not isinstance(main_curve, rates.Curve):
        raise TypeError()
    if pmnt_date < cal_instr.start_date:
        raise ValueError()

    dcfs = cal_instr.day_count_fractions
    pmnt_dates = cal_instr.payment_dates
    fixing_date = cal_instr.start_date

    discounts = main_curve.get_dsc(pmnt_dates)
    annuity = np.dot(discounts, dcfs)
    betas = get_b(mr, fixing_date, pmnt_dates)
    gamma = np.dot(betas, dcfs * discounts) / annuity

    if fwd_curve is not None:
        flt_adjs = cal_instr.get_flt_adjustments(main_curve, fwd_curve)
    else:
        flt_adjs = np.zeros_like(cal_instr.payment_dates)

    fwd = cal_instr.get_swap_rate(main_curve, flt_adjs=flt_adjs)

    denom = main_curve.get_dsc(pmnt_date) * (gamma - get_b(mr, fixing_date, pmnt_date))

    numer = discounts[-1] * betas[-1] \
        - np.dot(flt_adjs, dcfs * discounts * betas) + annuity * fwd * gamma
    a = denom / numer
    b = main_curve.get_dsc(pmnt_date) / annuity - a * fwd
    return a, b


def get_impl_mr(tgt_a, cal_instr, main_curve, pmnt_date: float, fwd_curve=None):
    def func(mr):
        res = get_linear_coeffs(cal_instr, main_curve, mr, pmnt_date, fwd_curve)[0]
        return np.square(res - tgt_a)

    return optimize.root(func, [0.01]).x[0]
