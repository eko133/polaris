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


dbe =dict()
basket = pd.DataFrame()
with open (r'./negative_ESI_result.pkl','rb') as f:
    data=pickle.load(f)
for i in data:
    data[i] = data[i][data[i]['Class'] == 'N1']
    # for m in range(9,20):
    #     tmp2 = data[i][data[i]['dbe'] == m]
    #     c_min = tmp2['C'].min()
    #     c_mean = tmp2['C'].mean()
    #     c_max = tmp2['C'].max()
    #     tmp2_upper = tmp2[(tmp2['C'] >= c_mean) & (tmp2['C'] >= c_mean)<=c_max]
    #     tmp2_lower = tmp2[(tmp2['C'] >= c_min) & (tmp2['C'] >= c_mean)<=c_mean]
    #

    dbe[i] = data[i].groupby('dbe').agg({'I':'sum'})
    dbe[i][i] = dbe[i].I/dbe[i].I.sum()
    del dbe[i]['I']
    basket = basket.merge(dbe[i],how='outer',right_index=True,left_index=True)
tmp1 = basket.copy()

tmp1 = tmp1.T

tmp1 = tmp1.dropna(axis=1)
tmp1['sample'] = tmp1.index
tmp1[['bio,','therm,']]=tmp1['sample'].str.split('_',expand=True)
tmp1['bio,'] = tmp1['bio,'].str.replace('L','')
del tmp1['sample']

rdata = pd.DataFrame(columns={'Intercept', 'Coef', 'Score'})
for x, y in combinations(tmp1.columns, 2):
    if (x!='bio,') & (x!='therm,') & (y!='bio,') & (y!='therm,'):
        tmp1['%s,%s'%(x,y)] = tmp1[x]/tmp1[y]
for column in tmp1.columns:
    if ',' not in column:
        del tmp1[column]
for x in tmp1.columns:
    tmp = tmp1[[x, 'therm,']].copy()
    tmp = tmp.dropna(how='any', axis=0)
    x1 = tmp[x].values.reshape(-1, 1)
    y1 = tmp['therm,'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.1, random_state=0)
    LR = linear_model.LinearRegression()
    LR.fit(X_train, y_train)
    rdata.loc['%s, %s' % (x, 'therm,'), 'Intercept'] = LR.intercept_
    rdata.loc['%s, %s' % (x, 'therm,'), 'Coef'] = LR.coef_
    rdata.loc['%s, %s' % (x, 'therm,'), 'Score'] = LR.score(X_test, y_test)
rdata.to_csv('./rdata.csv')