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

basket = dict()
with open (r'./negative_ESI_result.pkl','rb') as f:
    data = pickle.load(f)
class_abundance=dict()
orig_sample = {'L0':0,'L2':0,'L5':0,'L8':0}
for i in data:
    data[i] =data[i][(data[i]['Class']!='') & (data[i]['Class']!='O4') & (data[i]['Class']!='O4N1') &  (data[i]['Class']!='O3')]
    class_abundance[i] = data[i].groupby('Class').agg({'I':'sum'})
    class_abundance[i][i] = class_abundance[i].I/class_abundance[i].I.sum()
    del class_abundance[i]['I']
for i in orig_sample:
    tmp = {key: value for key, value in class_abundance.items() if i in key}
    basket[i] = pd.DataFrame()
    for key in tmp:
        basket[i] = basket[i].merge(tmp[key], how='outer', left_index=True,right_index=True)
    basket[i].to_csv(r'./data/%s_class_abundance.csv' % i)




