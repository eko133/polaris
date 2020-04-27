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
with open (r'./pkl/negative_ESI_result.pkl','rb') as f:
    data = pickle.load(f)
for i in data:
    data[i] = data[i].dropna()

    data[i]['C'] = data[i]['C'].astype(int)
    data[i]['I'] = data[i]['I'].astype(float)
    tmp = tmp[['em','I','Class','dbe','C']]
    tmp.em = tmp.em.astype(str)
    tmp.dbe = tmp.dbe.astype(str)
    tmp.C = tmp.C.astype(str)
    tmp.em = tmp.em + ','+ tmp.Class+','+ tmp.dbe+','+ tmp.C
    del tmp['Class']
    del tmp['dbe']
    del tmp['C']
    tmp['I'] = (tmp['I'] - tmp['I'].min())/(tmp['I'].max() - tmp['I'].min())
    tmp = tmp.rename(columns={'I':i})
    tmp = tmp.set_index('em')
    pca = pca.merge(tmp,how='outer',left_index=True,right_index=True)
pca = pca.replace(np.nan,0)
pca=pca.T

pc = PCA(n_components=2)
pComponents = pc.fit_transform(pca)
print(pc.explained_variance_ratio_)
pcaData = pd.DataFrame(data=pComponents, columns=['PC 1', 'PC 2'])
pcaData.index = pca.index
loadings = pc.components_
loadings = loadings.T
loadings = pd.DataFrame(loadings)
loadings.index = pca.T.index

pcaData['sample'] = pcaData.index

pcaData['sample'], pcaData['temp']  = pcaData['sample'].str.split('_', 1).str
plt.scatter(pcaData['PC 1'],pcaData['PC 2'])
plt.show()

loadings.to_csv( r'./data/pca/loadings.csv')


pcaData.to_csv(r'./data/pca/pca_results.csv')


