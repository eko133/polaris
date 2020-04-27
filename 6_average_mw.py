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
with open (r'./pkl/negative_ESI_result.pkl','rb') as f:
    data=pickle.load(f)
for i in data:
    data[i] = data[i][(data[i]['Class'] != '') & (data[i]['Class'] != 'O4') & (data[i]['Class'] != 'O4N1')&  (data[i]['Class']!='O3')]
    tmp = data[i]
    tmp=tmp.dropna()
    tmp2 = tmp[['I','em']]
    tmp2['mw1'] = tmp2['I'] * tmp2['em']
    tmp2['mn1'] = tmp['em']*tmp['em']*tmp['I']

    avr.loc[i,'mw'] = tmp2['mw1'].sum()/tmp['I'].sum()
    avr.loc[i,'mn'] = tmp2['mn1'].sum()/tmp2['mw1'].sum()
    avr.loc[i,'signal'] = len(tmp2)
avr.to_csv(r'./average_mass1.csv')
