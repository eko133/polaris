from execjs import get
import os
import pandas as pd
import numpy as np
import pickle

# mass_list=set()
# pcatable = pd.DataFrame()
# runtime = get('Node')
# context = runtime.compile('''
#     module.paths.push('%s');
#     chemcalc = require('chemcalc');
#     function mfFromMonoisotopicMass(mass,cfg){
#           return chemcalc.mfFromMonoisotopicMass(mass,cfg);
#     }
# ''' % os.path.join(os.path.dirname(__file__),'node_modules'))
#
# f = open(r'/Users/siaga/line_test.txt','r')
# lines =f.readlines()
# samples = {}
# samples_extra = {}
# basket = pd.DataFrame()
# for line in lines:
#     data = line.split(';')
#     sample_name = data[0]
#     samples[sample_name] = pd.DataFrame()
#     del data[0]
#     del data[1]
#     data = pd.DataFrame(np.array(data).reshape((-1,3)),columns=['m/z','I','S/N'])
#     data = data.astype(float)
#     raw_mass_intensity = data['I'].sum()
#     for i in range(len(data)):
#         if i < len(data):
#             mass_Na = data.loc[i, 'm/z']
#             result = context.call('mfFromMonoisotopicMass',mass_Na,{'mfRange':'C1-200H1-200O0-10N0-10S0-5Na+','maxUnsaturation':'10','useUnsaturation':'true','integerUnsaturation':'false','massRange':'0.01'})
#             try:
#                 for m in range(len(result['results'])):
#                     if not isinstance(result['results'][m]['unsat'], int):
#                     # should the carbon isotopes be tested?
#                         mass_iso_em = result['results'][m]['em'] - 12 + 13.003355
#                         mass_iso = data[(data['m/z'] >= (mass_iso_em - 0.005)) & (data['m/z'] <= (mass_iso_em + 0.005))]
#                         data = data[data['m/z'] != mass_iso['m/z'].max()].reset_index(drop=True)
#                         data.loc[i, 'real mass'] = result['results'][m]['em']
#                         data.loc[i, 'error(ppm)'] = result['results'][m]['ppm']
#                         data.loc[i, 'mf'] = result['results'][m]['mf']
#                         data.loc[i, 'unsat'] = result['results'][m]['unsat']
#                         break
#             except IndexError:
#                 data.loc[i,'real mass'] = 'nan'
#         else:
#             break
#     if 'real mass' in data.columns:
#         data= data.dropna(axis=0).reset_index(drop=True)
#         data = data[(data['error(ppm)'] < 10) & (data['error(ppm)'] > -10)]
#         data = data[data['m/z'] > 1000]
#         samples[sample_name] = data.copy()
#         processed_mass_intensity = data['I'].sum()
#         print(processed_mass_intensity/raw_mass_intensity)
#         data['m/z'] = data['real mass'].astype(str) + ',' + data['mf']
#         for column in data:
#             if column != 'm/z' and column != 'I':
#                 del data[column]
#         mass_list.update(data['m/z'])
#         data = data.rename(columns={'m/z': 'mass%s'%sample_name,'I':sample_name})
#         pcatable = pd.concat([pcatable, data], axis=1, sort=False)
#
# pickle.dump(samples, open('./samples_line_test.p', 'wb'))
#
# for key in samples:
#     samples_extra[key] = pcatable[['mass' + key, key]].dropna()
#     samples_extra['mass'+key] = mass_list - set(pcatable['mass' + key])
#     samples_extra['mass' + key] = pd.DataFrame(samples_extra['mass' + key], columns=['mass' + key])
#     samples_extra[key] = pd.concat([samples_extra[key], samples_extra['mass' + key]], ignore_index=True,
#                                      sort=False).fillna(0)
#     samples_extra[key] = samples_extra[key].sort_values(by=['mass' + key]).reset_index(drop=True)
#     basket = pd.concat([basket, samples_extra[key]], axis=1, sort=False)
# basket = basket.rename(columns={basket.columns[0]:'mtoz'})
# for column in basket:
#      if 'mass' in column:
#         del basket[column]
# basket[['mtoz', 'formula']] = basket['mtoz'].str.split(',', expand=True)
# basket['mtoz'] = basket['mtoz'].astype(float)
# basket.to_pickle("./line_test.pkl")

basket = pd.read_pickle("./line_test.pkl")
similar_index=list()
mass_index=0
duplicate_flag = 1
while duplicate_flag == 1:
    duplicate_flag = 0
    while mass_index < (len(basket)-1):
        if (basket.loc[mass_index+1,'mtoz'] - basket.loc[mass_index,'mtoz']) < 0.02:
            duplicate_flag = 1
            mass_tmp = basket.loc[mass_index, 'mtoz']
            basket.loc[mass_index] = basket.loc[mass_index] + basket.loc[mass_index + 1]
            for column in basket:
                if 'ROOX' in column:
                    basket.loc[mass_index+1,column] = 0
            basket.loc[mass_index, 'mtoz'] = mass_tmp
            similar_index.append(mass_index)
        mass_index = mass_index + 1
    for i in reversed(similar_index):
        basket=basket.drop(basket.index[i+1])
    basket = basket.reset_index(drop=True)
    mas_index = 0
    similar_index=list()

basket.to_pickle("./gdgt_similarMassMerged.pkl")

