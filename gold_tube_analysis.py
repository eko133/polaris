from execjs import get
import pandas as pd
import numpy as np
import sys
import os
import crude_oil
from multiprocessing import cpu_count, Pool
from functools import partial
import ast
import pickle



def parallelize(data, func):
    data_split = np.array_split(data, cores)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def chemcaljs_column(data):
    data['result'] = data['m/z'].apply(crude_oil.chemcaljs)
    return data


if __name__ == '__main__':
    data_cluster = crude_oil.read_raw_csv()
    processed_data = dict()
    cores = cpu_count()
    for key in data_cluster:
        data = data_cluster[key]
        data = parallelize(data, chemcaljs_column)
        data = data.dropna()
        # data['result'] = data['result'].apply(ast.literal_eval)
        data = crude_oil.extract_result(data)
        data = crude_oil.extract_mf(data)
        data = crude_oil.custom_data_filter1(data)
        processed_data[key] = data.copy()
    with open ('./processed_data.pickle') as f:
        pickle.dump(processed_data,f)




    # for key in data_cluster:
    #     data = data_cluster[key]
    #     data = parallelize(data, chemcaljs_column)
    #     data = data[data['em']!=None]


# for key in data_cluster:
#
# for i in range(len(data)):
#     if i < len(data):
#         mass_Na = data.loc[i, 'm/z']
#         result = context.call('mfFromMonoisotopicMass',mass_Na,{'mfRange':'C1-200H1-200O0-10Na+','maxUnsaturation':'10','useUnsaturation':'true','integerUnsaturation':'false','massRange':'0.006'})
#         try:
#             for m in range(len(result['results'])):
#                 if not isinstance(result['results'][m]['unsat'], int):
#                 # should the carbon isotopes be tested?
#                     mass_iso_em = result['results'][m]['em'] - 12 + 13.003355
#                     mass_iso = data[(data['m/z'] >= (mass_iso_em - 0.005)) & (data['m/z'] <= (mass_iso_em + 0.005))]
#                     data = data[data['m/z'] != mass_iso['m/z'].max()].reset_index(drop=True)
#                     data.loc[i, 'real mass'] = result['results'][m]['em']
#                     data.loc[i, 'error(ppm)'] = result['results'][m]['ppm']
#                     data.loc[i, 'mf'] = result['results'][m]['mf']
#                     data.loc[i, 'unsat'] = result['results'][m]['unsat']
#                     break
#         except IndexError:
#             data.loc[i,'real mass'] = 'nan'
#     else:
#         break
# if 'real mass' in data.columns:
#     data= data.dropna(axis=0).reset_index(drop=True)
#     data = data[(data['error(ppm)'] < 10) & (data['error(ppm)'] > -10)]
#     samples[sample_name] = data.copy()
#     data['m/z'] = data['real mass'].astype(str) + ',' + data['mf']
#     for column in data:
#         if column != 'm/z' and column != 'I':
#             del data[column]
#     mass_list.update(data['m/z'])
#     data = data.rename(columns={'m/z': 'mass%s'%sample_name,'I':sample_name})
#     pcatable = pd.concat([pcatable, data], axis=1, sort=False)
#
# pickle.dump(samples, open('./samples.p', 'wb'))
#
