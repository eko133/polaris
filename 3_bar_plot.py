import pandas as pd
from mendeleev import element
import itertools
import run
import numpy as np
import sys
import os
from multiprocessing import cpu_count, Pool
from functools import partial
import ast
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

contaminants = (['O2',1,16],['O2',1,18],['O2',2,18],['O3',2,18],['O3',3,18],['O4',2,18],['O4',3,18],['O4',4,18])
basket = dict()
with open (r'./pkl/negative_ESI_result_2.pkl','rb') as f:
    data = pickle.load(f)
class_abundance=dict()
orig_sample = {'L0':0,'L2':0,'L5':0,'L8':0}
for i in data:
    # data[i] =data[i][(data[i]['Class']!='') | (data[i]['Class']!='O4') | (data[i]['Class']!='O4N1') |  (data[i]['Class']!='O3')]
    data[i] =data[i][(data[i]['Class'] =='O2') | (data[i]['Class'] == 'O1Cl1') | (data[i]['Class'] == 'O3') | (data[i]['Class']=='O4') | (data[i]['Class'] =='N1Cl1') | (data[i]['Class']=='O2N1') |  (data[i]['Class']=='O1N1') | (data[i]['Class'] == 'O3N1')]
    data[i].dbe = data[i].dbe.astype(int)
    data[i].C = data[i].C.astype(int)
    for m in contaminants:
       data[i] = data[i].drop(data[i][(data[i].Class==m[0]) & (data[i].dbe==m[1]) & (data[i].C==m[2])].index)
    data[i] = data[i].drop(data[i][(data[i].Class=='O2N1') & (data[i].C >30)].index)
    data[i] = data[i].drop(data[i][(data[i].Class=='O3') & (data[i].C >25)].index)
    data[i] = data[i].drop(data[i][(data[i].Class=='O4') & (data[i].C >25)].index)
    data[i] = data[i].drop(data[i][(data[i].Class=='O3N1') & (data[i].C >25)].index)
    class_abundance[i] = data[i].groupby('Class').agg({'I':'sum'})
    class_abundance[i][i] = class_abundance[i].I/class_abundance[i].I.sum()
    del class_abundance[i]['I']
for i in orig_sample:
    tmp = {key: value for key, value in class_abundance.items() if i in key}
    basket[i] = pd.DataFrame()
    for key in tmp:
        basket[i] = basket[i].merge(tmp[key], how='outer', left_index=True,right_index=True)
    basket[i].to_csv(r'./data/%s_class_abundance.csv' % i)




