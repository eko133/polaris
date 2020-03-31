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
import matplotlib.pyplot as plt

pca = pd.DataFrame()
with open (r'./negative_ESI_result.pkl','rb') as f:
    data = pickle.load(f)
for i in data:
    data[i] =data[i][(data[i]['Class']!='') & (data[i]['Class']!='O4') & (data[i]['Class']!='O4N1')& (data[i]['Class']!='O3')]
    data[i] = data[i].dropna()
    data[i] = data[i][data[i]['Class']=='O2']
    # data[i]['dbe']  = data[i]['dbe'].astype(int)
    # data[i] = data[i][(data[i]['dbe']>=9) & (data[i]['dbe']<=15)]
    data[i] = data[i][['em','I','Class','dbe','C']]
    data[i].em = data[i].em.astype(str)
    data[i].dbe = data[i].dbe.astype(str)
    data[i].C = data[i].C.astype(str)
    data[i].em = data[i].em + ','+ data[i].Class+','+ data[i].dbe+','+ data[i].C
    del data[i]['Class']
    del data[i]['dbe']
    del data[i]['C']
    data[i]['I'] = (data[i]['I'] - data[i]['I'].min())/(data[i]['I'].max() - data[i]['I'].min())
    data[i] = data[i].rename(columns={'I':i})
    data[i] = data[i].set_index('em')
    pca = pca.merge(data[i],how='outer',left_index=True,right_index=True)
pca = pca.replace(np.nan,0)
# pca = pca.dropna(axis=0)
pca=pca.T

pc = PCA(n_components=2)
pComponents = pc.fit_transform(pca)
print(pc.explained_variance_ratio_)
pcaData = pd.DataFrame(data=pComponents, columns=['factor 1', 'factor 2'])
pcaData.index = pca.index
loadings = pc.components_
loadings = loadings.T
loadings = pd.DataFrame(loadings)
loadings.index = pca.T.index

pcaData['sample'] = pcaData.index

pcaData['sample'], pcaData['temp']  = pcaData['sample'].str.split('_', 1).str
plt.scatter(pcaData['factor 1'],pcaData['factor 2'])
plt.show()

loadings.to_csv( r'./O2_loadings.csv')


pcaData.to_csv(r'./O2_pca_results.csv')


