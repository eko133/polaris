import pandas as pd
import numpy as np
import sys
import os
from execjs import get
import re


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


def chemcaljs(m_to_z):
    print(m_to_z)
    runtime = get('Node')
    context = runtime.compile('''
        module.paths.push('%s');
        chemcalc = require('chemcalc');
        function mfFromMonoisotopicMass(mass,cfg){
              return chemcalc.mfFromMonoisotopicMass(mass,cfg);
        }
    ''' % os.path.join(os.path.dirname(__file__), 'node_modules'))

    mf_option = {'mfRange': 'C1-50H1-102O0-5N0-2Cl0-1',
                 'maxUnsaturation': '20',
                 'useUnsaturation': 'true',
                 'integerUnsaturation': 'false',
                 'massRange': '0.005'
                 }
    try:
        result = context.call('mfFromMonoisotopicMass', m_to_z, mf_option)['results'][0]
        # custom filter in the case of Cl contamination
        # if 'Cl' in result['mf']:
        #     if (result['unsat'] != int(result['unsat'])) or (re.findall(r'H(\d{1,9})', result['mf'])[0] % 2 != 0):
        #         return np.nan
        # else:
        #     if result['unsat'] == int(result['unsat']) or (re.findall(r'H(\d{1,9})',result['mf'])[0] % 2 == 0):
        #         return np.nan
        return result
        ##
    except IndexError:
        return np.nan

def extract_result(data):
    data['error'] = data['result'].apply(lambda x: x.get('error'))
    data['em'] = data['result'].apply(lambda x:x.get('em'))
    data['mf'] = data['result'].apply(lambda x:x.get('mf'))
    data['unsat'] = data['result'].apply(lambda x:x.get('unsat'))
    return data


def extract_mf(data):
    data['C'] = data['mf'].str.extract(r'C(\d{1,9})')
    data['H'] = data['mf'].str.extract(r'H(\d{1,9})')
    data['Class'] = data['mf'].str.replace(r'C\d{1,9}H\d{1,9}','')
    data = data.dropna(subset=['H'])
    return data


def custom_data_filter1(data):
    datacl = data[data['Class'].str.contains('Cl')]
    datacl = datacl[datacl.unsat % 1 == 0]

    data = data[~data['Class'].str.contains('Cl')]
    data = data[data.unsat % 1 != 0]
    # for negative mode
    data['unsat'] = data['unsat'].astype(float) - 0.5
    data['H'] = data['H'].astype(int) + 1

    data = pd.concat([datacl,data])
    return data


if __name__ == "__main__":
    main()
