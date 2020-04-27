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
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

pca=pd.DataFrame()
with open (r'./pkl/negative_ESI_result.pkl','rb') as f:
    data=pickle.load(f)
for i in data:
    data[i] =data[i][(data[i]['Class']!='') & (data[i]['Class']!='O4') & (data[i]['Class']!='O4N1')]
    data[i] = data[i].dropna()
    # data[i] = data[i][data[i]['Class'] == 'N1']
    data[i] = data[i][['em','I','Class']]
    data[i].em = data[i].em.astype(str)
    data[i].em = data[i].em + ','+ data[i]['Class']
    del data[i]['Class']
    data[i]['I'] = (data[i]['I'] - data[i]['I'].min())/(data[i]['I'].max() - data[i]['I'].min())
    data[i] = data[i].rename(columns={'I':i})
    data[i] = data[i].set_index('em')
    pca = pca.merge(data[i],how='outer',left_index=True,right_index=True)
pca = pca.replace(np.nan,0)
pca=pca.T
pca['sample'] = pca.index
pca[['bio','thermo']] = pca['sample'].str.split('_',expand=True)
del pca['sample']
del pca['thermo']

X=np.array(pca.drop(columns={'bio'}))
Y=np.array(pca['bio'])
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X,Y)
X_lda = lda.transform(X)
ldaData = pd.DataFrame(data=X_lda, columns=['principal component 1', 'principal component 2'])
ldaData.index = pca.index
loadings = lda.scalings_
# loadings = loadings.T
loadings = pd.DataFrame(loadings)
loadings.index = pca.drop(columns={'bio'}).T.index
loadings.to_csv( r'./loadings.csv')
ldaData.to_csv(r'./lda_results.csv')

