import xml.etree.ElementTree as ET

import numpy as np

from script.fox_toolbox.utils.rates import Curve, Swap, Swaption, Volatility


def get_xml(fpath):
    xml_tree = ET.parse(fpath)
    xml_root = xml_tree.getroot()
    return xml_tree, xml_root


def get_str_node(root_node, xpath, default=None):
    val = root_node.findtext(xpath)
    return val.strip() if val is not None else default


def get_float_node(root_node, xpath, default=None):
    val = get_str_node(root_node, xpath, None)
    return float(val) if val is not None else default


def get_int_node(root_node, xpath, default=None):
    val = get_str_node(root_node, xpath, None)
    return int(val) if val is not None else default


def get_delim_str_node(root_node, xpath, default=None):
    val = get_str_node(root_node, xpath, None)
    if val is None:
        return [default]
    return list(map(str.strip, val.split(';')))


def get_delim_float_node(root_node, xpath, default=None):
    return delim_node_to_array(root_node.find(xpath), default)


def delim_node_to_array(node, default=None):
    return np.array(node.text.split(';'), dtype=float) if node is not None else np.array([default], dtype=float)


def parse_1d_curve(node):
    if node is None:
        raise ValueError("Argument is null.")
    if node.tag != 'OneDCurve':
        raise ValueError("Not a 1d curve node.")

    label = get_str_node(node, 'CurveLabel', '')
    buckets = get_delim_float_node(node, 'Buckets')
    values = get_delim_float_node(node, 'Values')
    interp = get_str_node(node, 'InterpolationMethod')
    return label, interp, buckets, values


def parse_2d_curve(node):
    if node is None:
        raise ValueError("Argument is null.")
    if node.tag != 'TwoDCurve':
        raise ValueError("Not a 2d curve node.")

    interp = get_str_node(node, 'InterpolationMethod')
    buckets_x = get_delim_float_node(node, 'Buckets')
    buckets_y = get_delim_float_node(node, 'Values/OneDCurve/Buckets')
    values = np.array([delim_node_to_array(n) for n in node.findall('Values/OneDCurve/Values')])
    return interp, buckets_x, buckets_y, values


def get_rate_curve(curve_node):
    if curve_node is None:
        raise ValueError("Argument is null.")

    if curve_node.tag != 'OneDCurve':
        if curve_node[0].tag == 'OneDCurve':
            curve_node = curve_node[0]
        else:
            raise ValueError("Cannot find curve.")

    label, interp, buckets, values = parse_1d_curve(curve_node)
    curve = Curve(buckets, values, interp, label)
    return curve


def get_curves(xmlfile):
    main_curve = get_rate_curve(xmlfile.find('.//RateCurve'))
    sprd_curve_nodes = xmlfile.findall('.//SpreadRateCurveList/SpreadRateCurves/SpreadRateCurve/OneDCurve')
    sprd_curves = []
    for node in sprd_curve_nodes:
        label, interp, buckets, values = parse_1d_curve(node)
        curve = Curve(buckets, values + main_curve.zc_rates, interp, label)
        sprd_curves.append(curve)
    if len(sprd_curves) == 0:
        sprd_curves = None
    return main_curve, sprd_curves


def get_hw_params(xmlfile):
    params_node = xmlfile.find('.//FactorsList/Factors/Factor')
    _, _, buckets, values = parse_1d_curve(params_node.find('VolatilityCurve/OneDCurve'))
    mr = get_float_node(params_node, 'MeanRR')
    return mr, (buckets, values)
    

def get_calib_instr(node):
    if node is None:
        raise ValueError("Argument is null.")
    if node.tag != 'CalibrationInstrument':
        raise ValueError("Not a calibration instrument.")

    atm_vol = get_float_node(node, 'BlackVol')
    if atm_vol == 0.0:
        atm_vol = get_delim_float_node(node, 'SmileSlice/OneDCurve/Values')
    
    vol_nature = node.find('VolModel')
    if vol_nature is None:
        shift = get_float_node(node, 'SmileShift', 0.0)
    else:
        shift = None
    vol = Volatility(atm_vol, 'SLN' if vol_nature is None else 'N', shift)
    flows = [f.text.split(';') for f in node.findall('Flows/Flow')]
    fixing_date = float(flows[0][0])
    pmnt_dates = np.array([f[0] for f in flows[1:-1]], dtype=float)
    dcfs = np.array([f[2] for f in flows[1:-1]], dtype=float)

    return Swaption(
        get_float_node(node, 'OptionExpiry'),
        vol,
        fixing_date,
        pmnt_dates,
        dcfs,
        get_int_node(node, 'CalInstTenor'),
        lvl=get_int_node(node, 'CalibrationLevel'),
        cal_type=get_str_node(node, 'CalInstKType'),
		pay_rec=get_str_node(node, 'PayReceive'),
        strike=get_float_node(node, 'Strike', np.nan) / 100.,
        cal_vol=get_float_node(node, 'CalibratedVolatility', np.nan),
        fwd=get_float_node(node, 'SwapRateInfo/SpotSwapRate', np.nan),
        annuity=get_float_node(node, 'SwapRateInfo/FixLeg', np.nan),
        tgt_premium=get_float_node(node, 'BlackPrice', np.nan),
        cal_premium=get_float_node(node, 'CalibratedPremium', np.nan),
        fwdAdj=get_float_node(node, 'SwapRateInfo/FwdSwapRateCmsRep', np.nan),
        fwdAdjModel=get_float_node(node, 'SwapRateInfo/FwdSwapRateModel', np.nan),
    )


def get_calib_basket(xmlfile):
    for instr in xmlfile.iterfind(".//CalibrationInstrument"):
        yield get_calib_instr(instr)
