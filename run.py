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
        'Cl': 2,
        'DBE': 20
    }
    max_element_dict['H_max'] = 2 * max_element_dict['C'] + 2
    for c, h, o, n, cl in itertools.product(range(10, max_element_dict['C']), range(max_element_dict['H_max']), range(max_element_dict['O']), range(max_element_dict['N']), range(max_element_dict['Cl'])):
        dbe = c + 1 + n / 2 - h / 2 - cl/2
        if (0.3 <= h / c <= 3.0) & (o / c <= 3.0) & (n / c <= 0.5) & (0 <= dbe <=20) & (
                (h + n + cl) % 2 == 1):
            em = c * mono_isotope_mass_dict['C'] + h * mono_isotope_mass_dict['H'] + o * mono_isotope_mass_dict[
                'O'] + n * mono_isotope_mass_dict['N'] + cl * mono_isotope_mass_dict['Cl'] + 0.0005485799
            if 200 <= em <= 800:
                formula = 'C' + '%i'%c +'H' +'%i'%h +'O' +'%i'%o +'N' +'%i'%n + 'Cl' + '%i'%cl
                compound[em] = formula + ',' + str(int(dbe-0.5))
    with open (r'./dict/neg_esi_compound_dict.json','w') as f:
        json.dump(compound,f)
    return compound



def read_raw_csv():
    signal_to_noise_thresh = 6
    m_to_z_min = 200
    m_to_z_max = 800
    raw_data_path = r'C:\Users\siaga\OneDrive\Documents\黄金管FT\原始数据'
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
    data = data[(data['mf'] != 'C18H33O2N0Cl0') & (data['mf'] != 'C18H35O2N0Cl0') & (data['mf'] != 'C16H31O2N0Cl0')]
    data['C'] = data['mf'].str.extract(r'C(\d{1,9})')
    data['H'] = data['mf'].str.extract(r'H(\d{1,9})')
    data['Class'] = data['mf'].str.replace(r'C\d{1,9}H\d{1,9}', '')
    data['Class'] = data['Class'].str.replace(r'[a-zA-z]{1,9}0', '')
    del data['mf']
    return data


def csv_to_pkl(path):
    data = dict()
    raw_data = os.listdir(path)
    for csv in raw_data:
        if 'csv' in csv:
            data[os.path.splitext(csv)[0]] = pd.read_csv(path +os.sep+csv)
    with open(r'./negative_ESI_result.pkl', 'wb') as f:
        pickle.dump(data,f)


def filter_compounds(data):
    # data = data.drop(data[(data['Class'] == 'O2N1')&(data['C'] >=25)].index)
    # data = data.drop(data[(data['Class'] == 'O3')&(data['C'] >=30)].index)
    # data = data.drop(data[(data['Class'] == 'O3N1') & (data['C'] >= 25)].index)
    data = data.drop(data[(data['Class'] == 'O1N1') & (data['C'] >= 25)].index)
    return data


def carbazole_ternary(data):
    with open(data,'rb') as f:
        data = pickle.load(f)
    basket = pd.DataFrame()
    for i in data:
        tmp = data[i][data[i]['Class'] == 'N1Cl1']
        tmp['dbe'] = tmp['dbe'].astype(int)
        tmp['dbe'] = tmp['dbe']+1
        tmp['C'] = tmp['C'].astype(int)
        tmp['I'] = tmp['I'].astype(float)
        basket.loc[i, '9'] = tmp[tmp['dbe'] == 9]['I'].sum()
        basket.loc[i, '12'] = tmp[tmp['dbe'] == 12]['I'].sum()
        basket.loc[i, '15'] = tmp[tmp['dbe'] == 15]['I'].sum()
        basket['n9'] = 100 * basket['9'] / (basket['9'] + basket['12'] + basket['15'])
        basket['n12'] = 100 * basket['12'] / (basket['9'] + basket['12'] + basket['15'])
        basket['n15'] = 100 * basket['15'] / (basket['9'] + basket['12'] + basket['15'])
    return basket


def full_aromatized_to_partially_aromatized(data):
    with open(data,'rb') as f:
        data = pickle.load(f)
    basket = pd.DataFrame()
    for i in data:
        tmp = data[i][data[i]['Class'] == 'N1Cl1']
        tmp['dbe'] = tmp['dbe'].astype(int)
        tmp['dbe'] = tmp['dbe']+1
        tmp['C'] = tmp['C'].astype(int)
        tmp = tmp[tmp['C']<=55]
        tmp['I'] = tmp['I'].astype(float)
        tmp = tmp.drop(tmp[(tmp.dbe == 13) & (tmp.C == 33)].index)
        tmp = tmp.drop(tmp[(tmp.dbe == 14) & (tmp.C == 35)].index)
        full = tmp[(tmp['dbe'] == 9) | (tmp['dbe'] == 12) | (tmp['dbe'] == 15) | (tmp['dbe'] == 18) | (tmp['dbe'] == 21) | (tmp['dbe'] == 24)]['I'].sum()
        par = tmp[(tmp['dbe'] == 10) | (tmp['dbe'] == 11) | (tmp['dbe'] == 13) |(tmp['dbe'] == 14) | (tmp['dbe'] == 16) | (tmp['dbe'] == 17)|(tmp['dbe'] == 19) | (tmp['dbe'] == 20) | (tmp['dbe'] == 22) |(tmp['dbe'] == 23) | (tmp['dbe'] == 25) | (tmp['dbe'] == 26) ]['I'].sum()
        basket.loc[i, 'full_partial'] = full/par
    return basket


def carbon_number_distribution(data, specie,dbe):
    with open(data,'rb') as f:
        data = pickle.load(f)
    basket = pd.DataFrame()
    for i in data:
        data[i]['dbe'] = data[i]['dbe'].astype(int)
        data[i]['C'] = data[i]['C'].astype(int)
        data[i]['I'] = data[i]['I'].astype(float)
        tmp = data[i][(data[i]['Class'] == specie) & (data[i]['dbe'] == dbe) & (data[i]['C'] >= 10) & (data[i]['C'] <= 50)]
        tmp = tmp.drop(tmp[(tmp.dbe == 1) & (tmp.C == 16)].index)
        tmp = tmp.drop(tmp[(tmp.dbe == 1) & (tmp.C == 18)].index)
        tmp = tmp.drop(tmp[(tmp.dbe == 2) & (tmp.C == 18)].index)
        # tmp = data[i][(data[i]['Class'] == specie)  & (data[i]['C'] >= 10) & (data[i]['C'] <= 50)]
        for carbon in range(41):
        #     basket.loc[i,carbon] = tmp[tmp['C'] == carbon]['I'].sum()/tmp['I'].sum()
            basket.loc[i, carbon] = tmp[tmp['C'] == carbon]['I'].max()
    return basket


def dbe_distribution(data, specie):
    with open(data,'rb') as f:
        data = pickle.load(f)
    basket = pd.DataFrame()
    for i in data:
        data[i]['dbe'] = data[i]['dbe'].astype(int)
        data[i]['C'] = data[i]['C'].astype(int)
        data[i]['I'] = data[i]['I'].astype(float)
        tmp = data[i][(data[i]['Class'] == specie) & (data[i]['C'] >= 10) & (data[i]['C'] <= 40)]
        tmp = tmp.drop(tmp[(tmp.dbe == 1) & (tmp.C == 16)].index)
        tmp = tmp.drop(tmp[(tmp.dbe == 1) & (tmp.C == 18)].index)
        tmp = tmp.drop(tmp[(tmp.dbe == 2) & (tmp.C == 18)].index)
        for dbe in range(31):
            #basket.loc[i,carbon] = tmp[tmp['C'] == carbon]['I'].sum()/tmp['I'].sum()
            basket.loc[i, dbe] = tmp[tmp['dbe'] == dbe]['I'].sum()
    return basket

def cyclic_acids(data):
    with open(data,'rb') as f:
        data = pickle.load(f)
    basket = pd.DataFrame()
    for i in data:
        tmp = data[i][data[i]['Class'] == 'O2']
        tmp['dbe'] = tmp['dbe'].astype(int)
        tmp['C'] = tmp['C'].astype(int)
        tmp['I'] = tmp['I'].astype(float)

        tmp = tmp.drop(tmp[(tmp.dbe == 1) & (tmp.C == 16)].index)
        tmp = tmp.drop(tmp[(tmp.dbe == 1) & (tmp.C == 18)].index)
        tmp = tmp.drop(tmp[(tmp.dbe == 2) & (tmp.C == 18)].index)

        acyclic = tmp[tmp['dbe']==1]['I'].sum()

        cyclic = tmp[(tmp['dbe']>=2)&(tmp['dbe']<=6)]['I'].sum()
        cyclic2 = tmp[(tmp['dbe']>=7)&(tmp['dbe']<=20)]['I'].sum()
        basket.loc[i, 'O_ratio'] = acyclic/cyclic
        basket.loc[i, 'O_ratio2'] = cyclic/cyclic2
    return basket


def cpi(data):
    with open(data,'rb') as f:
        data = pickle.load(f)
    basket = pd.DataFrame()
    for i in data:
        tmp = data[i][data[i]['Class'] == 'O2']
        tmp['dbe'] = tmp['dbe'].astype(int)
        tmp['C'] = tmp['C'].astype(int)
        tmp['I'] = tmp['I'].astype(float)
        even = 0
        odd = 0
        for dbe in (1,2,3,4,5,6):
            for c in (22,24,26,28,30):
                even = even + tmp[(tmp['dbe']==dbe)&(tmp['C']==c)]['I'].max()
                odd = odd + tmp[(tmp['dbe']==dbe)&(tmp['C']==(c+1))]['I'].max()
            cpi = even/odd
            basket.loc[i, dbe] = cpi
    return basket


if __name__ == "__main__":
    main()