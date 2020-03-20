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
with open (r'./negative_ESI_result.pkl','rb') as f:
    data=pickle.load(f)
for i in data:
    data[i] =data[i][(data[i]['Class']!='') & (data[i]['Class']!='O4') & (data[i]['Class']!='O4N1')]
    data[i] = data[i][data[i]['Class'] == 'N1']
    data[i] = data[i][['em','I']]
    data[i]['I'] = (data[i]['I'] - data[i]['I'].min())/(data[i]['I'].max() - data[i]['I'].min())
    data[i] = data[i].rename(columns={'I':i})
    data[i] = data[i].set_index('em')
    pca = pca.merge(data[i],how='outer',left_index=True,right_index=True)
# pca = pca.replace(np.nan,0)
pca = pca.dropna(axis=0)
pca=pca.T
pca['sample'] = pca.index
pca[['bio','thermo']] = pca['sample'].str.split('_',expand=True)
pca['thermo'] = pca['thermo'].astype(int)
pca['bio'] = pca['bio'].str.replace('L5','L2')
pca['bio'] = pca['bio'].str.replace('L8','L2')
pca = pca[pca['thermo'] <400]
del pca['thermo']
del pca['sample']

X=np.array(pca.drop(columns='bio'))
Y=np.array(pca['bio'])
X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.1)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train,Y_train)
score = lda.score(X_test,Y_test)
print(score)
# X_new = lda.transform(X)
# ldaData = pd.DataFrame(data=X_new, columns=['lda1', 'lda2'])
# ldaData.index = pca.index
# ldaData.to_csv(r'./lda.csv')

# pc = PCA(n_components=2)
# pComponents = pc.fit_transform(pca)
# print(pc.explained_variance_ratio_)
# pcaData = pd.DataFrame(data=pComponents, columns=['principal component 1', 'principal component 2'])
# pcaData.index = pca.index
# loadings = pc.components_
# loadings = loadings.T
# loadings = pd.DataFrame(loadings)
# loadings.index = pca.T.index
# loadings.to_csv( r'./loadings.csv')
# pcaData.to_csv(r'./pca_results.csv')

