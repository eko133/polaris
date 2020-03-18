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

lreg = pd.DataFrame()
basket = pd.DataFrame()
with open (r'./negative_ESI_result.pkl','rb') as f:
    data=pickle.load(f)
for i in data:
    tmp = data[i][data[i]['Class'] == 'N1']
    tmp['dbe'] = tmp['dbe'].astype(int)
    tmp['C'] = tmp['C'].astype(int)
    for dbe in [9,12,15,18]:
        tmp1 = tmp[tmp['dbe'] == dbe]
        tmp1 = tmp1[tmp1['C']>=15]
        c = tmp1['C'].min()
        basket.loc[dbe,'C'] = c
    basket = basket.dropna()
    y1 = basket.index.values.reshape(-1, 1)
    x1 = basket['C'].values.reshape(-1, 1)
    # X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.1, random_state=0)
    LR = linear_model.LinearRegression()
    LR.fit(x1, y1)
    lreg.loc['%s' %i, 'Intercept'] = LR.intercept_
    lreg.loc['%s' %i, 'Coef'] = LR.coef_

