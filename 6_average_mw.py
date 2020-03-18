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
from sklearn import linear_model
from itertools import combinations

avr=pd.DataFrame()
with open (r'./negative_ESI_result.pkl','rb') as f:
    data=pickle.load(f)
for i in data:
    data[i] = data[i][(data[i]['Class'] != '') & (data[i]['Class'] != 'O4') & (data[i]['Class'] != 'O4N1')]
    tmp = data[i]
    tmp = tmp[(tmp['Class'] != '')]
    tmp2 = tmp[['I','em']]
    tmp2['normalized'] = tmp2.I / tmp2.I.sum()
    tmp2['avrem'] = tmp2['normalized'] * tmp2['em']
    avr.loc[i,'total'] = tmp2['avrem'].sum()

    species = set(tmp['Class'])
    for m in species:
        tmp1 = tmp[tmp['Class'] == m]
        tmp1 = tmp1[['I','em']]
        tmp1['normalized'] = tmp1.I/tmp1.I.sum()
        tmp1['avrem'] = tmp1['normalized']*tmp1['em']
        avr.loc[i, m]= tmp1['avrem'].sum()
avr.to_csv(r'./average_mass1.csv')
