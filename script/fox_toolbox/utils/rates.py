import numpy as np
from collections import namedtuple

from scipy.interpolate import interp1d


def cast_to_array(x, type_=float):
    return np.array(x, dtype=type_)


def build_class_str(self, args_dic):
    def generate():
        yield type(self).__name__
        yield '-' * 80
        yield from (f'{key}: {val!r}' for key, val in args_dic)
    return '\n'.join(generate())


class Curve:
    """
    An interest rate curve object which build interpolator upon intialization
    and provides vectorized methods for efficient retrieval of zero-coupons and
    discount factors.

    Warning: Modification of curve pillars or values won't result in interpolator
    recalibration.
    """
    @staticmethod
    def build_curve(x, y):
        """
        Returns curve interpolator function
        """
        return interp1d(x, y, kind='linear', copy=True, bounds_error=False,
                        fill_value='extrapolate', assume_sorted=False)

    def __init__(self, dates: np.array, values: np.array, interpolation_method: str, label: str = ''):
        """
        Build a new curve
        :param dates: curve pillars as float array
        :param values: zero-coupon rates as float array
        :param interpolation_method: supporting only Linear and RateTime_Linear methods
        """
        self._interpolation_method = str(interpolation_method)
        self._dates = cast_to_array(dates, float)
        if np.any(self._dates < 0.):
            raise Exception('Negative dates are not supported')
        self._values = cast_to_array(values, float)
        # x = np.insert(self._dates, 0, 1e-6)
        # y = np.insert(self._values, 0, self._values[0])
        if interpolation_method == 'Linear':
            self._curve = Curve.build_curve(self._dates, self._values)
            self.__ratelinear = False
        elif interpolation_method == 'RateTime_Linear':
            self._curve = Curve.build_curve(self._dates, self._dates * self._values)
            self.__ratelinear = True
        else:
            raise NotImplementedError(f'"{self._interpolation_method}" interpolation method is not supported.')

        self._label = str(label)

    @property
    def interpolation_mode(self):
        return self._interpolation_method

    @property
    def label(self):
        return self._label

    @property
    def curve_pillars(self):
        return np.copy(self._dates)

    @property
    def zc_rates(self):
        return np.copy(self._values)

    def __iter__(self):
        return (i for i in zip(self._dates, self._values))

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}({self.label!r}, {self._dates!r}, {self._values!r}, {self.interpolation_mode})'

    def __str__(self):
        lbls = 'Name Pillars Zero-coupons Interpolation'.split(' ')
        data = (self.label, self._dates, self._values, self._interpolation_method,)
        return build_class_str(self, zip(lbls, data))

    def dump(self):
        return {
            'label': self.label,
            'pillars': list(self.curve_pillars),
            'zc_rates': list(self.zc_rates)
        }

    def get_zc(self, t):
        time = np.asarray(t)
        res = self._curve(time)
        if self.__ratelinear:
            res = np.divide(res, time, where=time > 0.,
                            out=np.full_like(time, np.nan, dtype=np.double))
        else:
            res[time < 0.] = np.nan
        res[time == 0.] = self._values[0]
        return res if res.size > 1 or isinstance(t, np.ndarray) else type(t)(res)

    def get_dsc(self, t):
        time = np.asarray(t)
        res = np.exp(-np.multiply(self.get_zc(time), time, where=time >= 0.,
                                  out=np.full_like(time, np.nan, dtype=np.double)))
        return res if res.size > 1 or isinstance(t, np.ndarray) else type(t)(res)

    def get_fwd_dsc(self, t: float, T):
        res = self.get_dsc(np.asarray(T)) / self.get_dsc(t)
        return res if res.size > 1 or isinstance(T, np.ndarray) else type(T)(res)


class Swap:

    def __init__(self, start_date, pmnt_dates, dcfs, libor_tenor):
        self._start_date = float(start_date)
        self._pmnt_dates = cast_to_array(pmnt_dates, float)
        self._dcfs = cast_to_array(dcfs, float)
        self._libor_tenor = int(libor_tenor)
        if self._pmnt_dates.size != self._dcfs.size:
            raise ValueError('Payment dates and day count fractions must be of same size.')

    @property
    def start_date(self):
        return self._start_date

    @property
    def payment_dates(self):
        return np.copy(self._pmnt_dates)

    @property
    def day_count_fractions(self):
        return np.copy(self._dcfs)

    @property
    def dcfs(self):
        return np.copy(self._dcfs)

    @property
    def libor_tenor(self):
        return self._libor_tenor

    @property
    def maturity(self):
        return self._pmnt_dates[-1]

    @property
    def swap_tenor(self):
        return self.maturity - self._start_date

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}({self._start_date!r}, {self._pmnt_dates!r}, {self._dcfs!r}, {self._libor_tenor})'

    def __str__(self):
        lbls = ('Start date', 'Payment dates', 'Day count fractions', 'Libor tenor',)
        data = (self._start_date, self._pmnt_dates, self._dcfs, self._libor_tenor)
        return build_class_str(self, zip(lbls, data))

    def get_annuity(self, dsc_curve: Curve):
        return np.dot(self._dcfs, dsc_curve.get_dsc(self._pmnt_dates))

    def get_flt_adjustments(self, dsc_curve: Curve, fwd_curve: Curve):
        pmnts_count = self._pmnt_dates.size
        last_period_length = self._pmnt_dates[-1] - self._pmnt_dates[-2] if pmnts_count > 1 else self.swap_tenor
        last_period_length_month = int(last_period_length * 12 + 0.5)
        frequency = max(int(last_period_length_month / self._libor_tenor), 1)

        flt_adjs = np.zeros(pmnts_count)
        for i in range(pmnts_count):
            t_start = self._start_date if i == 0 else self._pmnt_dates[i - 1]
            t_end = self._pmnt_dates[i]
            flt_adj = 0.0
            for j in range(frequency):
                ts = t_start + j / frequency * (t_end - t_start)
                te = t_start + (j + 1) / frequency * (t_end - t_start)
                flt_adj += 1.0 / dsc_curve.get_fwd_dsc(te, t_end) * (
                        1.0 / fwd_curve.get_fwd_dsc(ts, te) - 1.0 / dsc_curve.get_fwd_dsc(ts, te))
            flt_adj /= self._dcfs[i]
            flt_adjs[i] = flt_adj
        return flt_adjs

    def get_swap_rate(self, dsc_curve: Curve, fwd_curve: Curve=None, flt_adjs=None):
        dscs = dsc_curve.get_dsc(self._pmnt_dates)
        flt_leg = dsc_curve.get_dsc(self.start_date) - dscs[-1]
        annuity = np.dot(self._dcfs, dscs)
        if flt_adjs is None and fwd_curve is not None:
            flt_adjs = self.get_flt_adjustments(dsc_curve, fwd_curve)
        if flt_adjs is not None:
            flt_leg += np.dot(flt_adjs * self._dcfs, dscs)
        return flt_leg / annuity


Volatility = namedtuple('Volatility', 'value type shift_size')


class Swaption(Swap):
    def __init__(self, expiry, vol, start_date, pmnt_dates, dcfs, libor_tenor, **kwargs):
        super().__init__(start_date, pmnt_dates, dcfs, libor_tenor)
        self._expiry = float(expiry)
        if not isinstance(vol, Volatility):
            TypeError('{} must be a {}'.format('vol', Volatility))
        self._vol = vol

        for name, value in kwargs.items():
            if name in self.__dict__:
                raise KeyError(f'Class already contains definition of {name}')
            setattr(self, name, value)

    @property
    def expiry(self):
        return self._expiry

    @property
    def vol(self):
        return self._vol

    def __repr__(self):
        return f'{self.__dict__!r}'
