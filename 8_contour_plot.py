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
from scipy.stats import kde


with open (r'./negative_ESI_result_1.pkl','rb') as f:
    data=pickle.load(f)
for i in data:
    data[i] = data[i][data[i]['Class'] == 'O2']
    tmp = data[i][['C','dbe','I']]
    tmp['dbe'] = tmp['dbe'].astype(int)
    # tmp['dbe'] = tmp['dbe'] +1
    tmp['C'] = tmp['C'].astype(int)
    tmp = tmp.drop(tmp[(tmp.dbe == 1) & (tmp.C == 16)].index)
    tmp = tmp.drop(tmp[(tmp.dbe == 1) & (tmp.C == 18)].index)
    tmp = tmp.drop(tmp[(tmp.dbe == 2) & (tmp.C == 18)].index)
    tmp = tmp[(tmp['dbe'] >=1) & (tmp['dbe'] <=30)]
    tmp = tmp[(tmp['C'] >10) & (tmp['C'] <=55)]
    tmp['normalized'] = (tmp['I']-tmp['I'].min())/(tmp['I'].max()-tmp['I'].min())
    del tmp['I']
    x=tmp['C'].values
    x=np.array(x,dtype=float)
    y=tmp['dbe'].values
    y=np.array(y,dtype=float)
    z=tmp['normalized'].values
    z=np.array(z,dtype=float)
    plt.figure(figsize=(6,5),dpi=300)
    plt.scatter(x,y,s=150*z)
    plt.xticks(range(10,56,5), fontsize = 16)
    plt.yticks(range(1,31,5), fontsize = 16)
    # plt.title(i, fontsize = 20)
    plt.savefig(r'C:\Users\siaga\OneDrive\Documents\黄金管FT\气泡图\O2\%s'%i)


