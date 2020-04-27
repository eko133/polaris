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
import matplotlib.pyplot as plt

basket = pd.DataFrame()
with open (r'./negative_ESI_result.pkl','rb') as f:
    data=pickle.load(f)
for i in data:
    ratio = pd.DataFrame()
    data[i] = data[i].dropna()
    tmp = data[i][data[i]['Class'] == 'O2']
    tmp['C'] = tmp['C'].astype(int)
    tmp['H'] = tmp['H'].astype(int)
    tmp['O/C'] = 2/tmp['C']
    tmp['H/C'] = (tmp['H']+1)/tmp['C']
    tmp = tmp[['O/C','H/C']]
    tmp = tmp.rename(columns={'O/C':'%s_O'%i,'H/C':'%s_H'%i})
    basket = basket.merge(tmp, how='outer',right_index=True,left_index=True)
basket.to_csv(r'./O_ratio.csv')
