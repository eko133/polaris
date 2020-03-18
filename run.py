import pandas as pd
import numpy as np
import sys
import os
from execjs import get
import re
from mendeleev import element
import itertools
import json
import pickle


def generate_possible_formula():
    compound = dict()
    mono_isotope_mass_dict = {'C': 0, 'H': 0, 'O': 0, 'N': 0, 'S': 0, 'Cl': 0}
    for isotope in mono_isotope_mass_dict:
        mono_isotope_mass_dict[isotope] = element(isotope).isotopes[0].mass
    max_element_dict = {
        'C': 60,
        'O': 5,
        'N': 2,
        'DBE': 20
    }
    max_element_dict['H_max'] = 2 * max_element_dict['C'] + 2
    for c, h, o, n in itertools.product(range(10, max_element_dict['C']), range(max_element_dict['H_max']),
                                        range(max_element_dict['O']), range(max_element_dict['N'])):
        dbe = c + 1 + n / 2 - h / 2
        if (0.3 <= h / c <= 3.0) & (o / c <= 3.0) & (n / c <= 0.5) & (0 <= dbe <=20) & (
                (h + n) % 2 == 1):
            em = c * mono_isotope_mass_dict['C'] + h * mono_isotope_mass_dict['H'] + o * mono_isotope_mass_dict[
                'O'] + n * mono_isotope_mass_dict['N'] + 0.0005485799
            if 200 <= em <= 800:
                formula = 'C' + '%i'%c +'H' +'%i'%h +'O' +'%i'%o +'N' +'%i'%n
                compound[em] = formula + ',' + str(int(dbe-0.5))
    with open (r'./dict/neg_esi_compound_dict.json','w') as f:
        json.dump(compound,f)
    return compound



def read_raw_csv():
    signal_to_noise_thresh = 6
    m_to_z_min = 200
    m_to_z_max = 800
    raw_data_path = r'C:\Users\siaga\Desktop\duplicati\黄金管FT\原始数据'
    raw_data = os.listdir(raw_data_path)
    data = dict()
    for csv in raw_data:
        if 'csv' in csv:
            data[os.path.splitext(csv)[0]] = pd.read_csv(raw_data_path+os.sep+csv)
            data[os.path.splitext(csv)[0]] = data[os.path.splitext(csv)[0]][data[os.path.splitext(csv)[0]]['S/N'] >= signal_to_noise_thresh]
            data[os.path.splitext(csv)[0]] = data[os.path.splitext(csv)[0]][(data[os.path.splitext(csv)[0]]['m/z'] >= m_to_z_min) & (data[os.path.splitext(csv)[0]]['m/z'] <=m_to_z_max)]
    return data


def speculate_formula(data,compound):
    mass_tolerance = 0.001
    for mass in compound:
        data1 = data[(data['m/z'] <= float(mass) + mass_tolerance) & (data['m/z'] >= float(mass) - mass_tolerance)]
        if not data1.empty:
            print(mass)
            ind = data1[data1['I'] == data1['I'].max()].index.tolist()[0]
            data.loc[ind, 'em'] = mass
            data.loc[ind, 'dbe'] =compound[mass].split(',')[1]
            data.loc[ind, 'mf'] = compound[mass].split(',')[0]
            data.loc[ind, 'error'] = abs(float(mass) - data.loc[ind, 'm/z'])
    data.dropna(subset=['em'],inplace=True)
    return data

def extract_mf(data):
    data = data[(data['mf'] != 'C18H33O2N0') & (data['mf'] != 'C18H35O2N0') & (data['mf'] != 'C16H31O2N0')]
    data['C'] = data['mf'].str.extract(r'C(\d{1,9})')
    data['H'] = data['mf'].str.extract(r'H(\d{1,9})')
    data['Class'] = data['mf'].str.replace(r'C\d{1,9}H\d{1,9}', '')
    data['Class'] = data['Class'].str.replace(r'[a-zA-z]{1,9}0', '')
    del data['mf']
    return data


if __name__ == "__main__":
    main()