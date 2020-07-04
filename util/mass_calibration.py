import itertools
import numpy as np
import pandas as pd
from mendeleev import element


def generate_possible_formula(min_mass, max_mass):
    compound = dict()
    mono_isotope_mass_dict = dict()
    for i in ['C', 'H', 'O', 'N', 'S', 'Cl', 'Na', 'K']:
        mono_isotope_mass_dict[i] = element(i).isotopes[0].mass
    min_mass, max_mass = min_mass, max_mass
    max_element_dict = {
        'C': int(max_mass / mono_isotope_mass_dict['C']) + 1,
        'O': 6,
        'N': 6,
        'Na': 2,
        'K': 1,
        'DBE': 20
    }
    max_element_dict['H_max'] = 2 * max_element_dict['C'] + 3
    tmp = [range(max_element_dict[i]) for i in ['C', 'H_max', 'O', 'N', 'Na', 'K']]
    for c, h, o, n, na, k in itertools.product(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]):
        if (c >= 10) & ((na + k) == 1):
            dbe = c + 1 + n / 2 - h / 2
            if (0.3 <= h / c <= 3.0) & (o / c <= 3.0) & (n / c <= 0.5) & (dbe % 1 == 0) & (0 <= dbe <= 20) & (
                    (h + n) % 2 == 0):
                em = na * mono_isotope_mass_dict['Na']
                em = c * mono_isotope_mass_dict['C'] + h * mono_isotope_mass_dict['H'] + o * mono_isotope_mass_dict[
                    'O'] + n * mono_isotope_mass_dict['N'] + na * mono_isotope_mass_dict['Na'] + k * \
                     mono_isotope_mass_dict['K'] - 0.0005485799
                if min_mass <= em <= max_mass:
                    formula = f'C{c}H{h}O{o}N{n}Na{na}K{k}'
                    compound[em] = formula + ',' + str(dbe)
    return compound


def mp_align_with_recalibration(packed_args):
    def mass_correction(x, coef, intercept, mass_min, mass_max):
        if x >= mass_min and x <= mass_max:
            err = coef * x + intercept
            return x / (1 + err / 1000000)
        else:
            return x

    """
    First recalibrate the mass spectra with multiple known compounds, then align with theoratical m/z values
    lines = exported plain file data from Bruker DataAnalysis
    datan = DataFrame to collect aligned data
    mass_calibration_list = known molecules to calibrate the mass sepctra
    compound = possible chemical formulas within the mass range of the spectra
    """
    lines, datan, mass_calibration_list, compound = packed_args
    mass_calibration_list.sort()
    for line in lines:
        data = line.split(';')
        sample_name = data[0]
        print(sample_name)
        del data[0]
        del data[0]
        tmp = pd.DataFrame(np.array(data).reshape((-1, 3)), columns=['m/z', 'I', 'S/N'])
        tmp = tmp.drop(columns='S/N')
        tmp = tmp.astype(float)
        tmp_cali = tmp.copy()
        for mass in mass_calibration_list:
            mass_min, mass_max = mass - 0.005, mass + 0.005
            tmp1 = tmp[(tmp['m/z'] >= mass_min) & (tmp['m/z'] <= mass_max)]
            try:
                idx = tmp1.idxmax()[1]
                tmp_cali.loc[idx, 'em'] = mass
            except ValueError:
                continue
        if not 'em' in tmp_cali.columns:
            continue
        tmp_cali = tmp_cali.dropna()
        tmp_cali.loc[:, 'ppm'] = 1000000 * (tmp_cali.loc[:, 'm/z'] - tmp_cali.loc[:, 'em']) / tmp_cali.loc[:, 'em']
        tmp_cali.loc[:, 'em'] = tmp_cali.loc[:, 'em'].round(4)
        cali = dict(zip(tmp_cali['m/z'], tmp_cali['ppm']))
        mass_calibration_list = list(cali.keys())
        for i in range(len(mass_calibration_list) - 1):
            coef = (cali[mass_calibration_list[i + 1]] - cali[mass_calibration_list[i]]) / (
                    mass_calibration_list[i + 1] - mass_calibration_list[i])
            intercept = cali[mass_calibration_list[i]] - coef * mass_calibration_list[i]
            if i == 0:
                tmp['m/z'] = tmp['m/z'].apply(
                    lambda x: mass_correction(x, 0, cali[mass_calibration_list[i]], 0, mass_calibration_list[i]))
            elif i == (len(mass_calibration_list) - 2):
                tmp['m/z'] = tmp['m/z'].apply(
                    lambda x: mass_correction(x, 0, cali[mass_calibration_list[i + 1]], mass_calibration_list[i + 1],
                                              1000))
            tmp['m/z'] = tmp['m/z'].apply(
                lambda x: mass_correction(x, coef, intercept, mass_calibration_list[i], mass_calibration_list[i + 1]))
        for key in compound:
            key_min, key_max = key - 0.0025, key + 0.0025
            tmp1 = tmp[(tmp['m/z'] >= key_min) & (tmp['m/z'] <= key_max)]
            try:
                id = tmp1.idxmax()[1]
                tmp.loc[id, 'em'] = key
            except ValueError:
                continue
        tmp = tmp.dropna()
        tmp.loc[:, 'ppm'] = 1000000 * (tmp.loc[:, 'm/z'] - tmp.loc[:, 'em']) / tmp.loc[:, 'em']
        tmp.loc[:, 'em'] = tmp.loc[:, 'em'].round(4)
        tmp = tmp.drop(columns=['m/z', 'ppm'])
        tmp = tmp.rename(columns={'I':sample_name})
        tmp = tmp.set_index('em')
        datan = datan.merge(tmp, how='outer', left_index=True, right_index=True)
    return datan

