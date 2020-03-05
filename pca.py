import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt

data = pd.read_pickle('./gdgt_similarMassMerged_Y50.pkl')
data = data.set_index('m/z')
column_length = len(data.columns)
data = data.dropna(thresh=0.5*column_length)
row_length = len(data)
data = data.dropna(thresh=0.5*row_length, axis=1)
data = data.fillna(0)
for column in data.columns:
    data[column] = (data[column]-data[column].min())/(data[column].max()-data[column].min())
data = data.T
for column in data.columns:
    if data[column].mean()>=0.5:
        del data[column]
data = data.T
for column in data.columns:
    data[column] = (data[column]-data[column].min())/(data[column].max()-data[column].min())
data = data.T

pca=PCA(n_components=2)
pComponents=pca.fit_transform(data)
print(pca.explained_variance_ratio_)
pcaData=pd.DataFrame(data=pComponents,columns = ['principal component 1', 'principal component 2'])
pcaData.index=data.index
loadings=pca.components_
loadings=loadings.T
loadings=pd.DataFrame(loadings)
loadings.index=data.T.index
loadings.to_excel('/Users/siaga/Desktop/loadings.xlsx')
pcaData.to_excel('/Users/siaga/Desktop/pca_results.xlsx')
#


