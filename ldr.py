import pandas as pd
import targeting
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

tmp1 = pd.read_csv('/Users/siaga/Git/polaris/data/grouped_by_ccat/normalized_ccat0.5189.csv')

tmp2 = pd.read_csv('/Users/siaga/Git/polaris/data/grouped_by_ccat/normalized_ccat0.42421.csv')

del tmp1[tmp1.columns[0]]
tmp1[tmp1.columns[0]] =tmp1[tmp1.columns[0]] + '_0.5189'
tmp1 = tmp1.set_index(tmp1.columns[0])
tmp1=tmp1.T


del tmp2[tmp2.columns[0]]
tmp2[tmp2.columns[0]] =tmp2[tmp2.columns[0]] + '_0.42421'
tmp2 = tmp2.set_index(tmp2.columns[0])
tmp2=tmp2.T

tmp = tmp1.merge(tmp2, how ='outer', left_index=True, right_index=True)

tmp =tmp.T
X=tmp.copy()
for column in X.columns:
    if X[column].mean() >= 0.5:
        del X[column]
X = X.replace(np.nan,0)
tmp ['labels'] =tmp.index
new = tmp ['labels'].str.split("_", n = 1, expand = True)
tmp ['labels']=new[1]
y = new[1]

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)