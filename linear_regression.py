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

data['ccat'] = data[1314.23]/(data[1314.23]+data[1324.31])
data['ccat'] = data['ccat'].replace(1,np.nan)
data['ccat'] = data['ccat'].replace(0,np.nan)
data = data.dropna(how='any',axis=0)
rdata = pd.DataFrame(columns={'Intercept','Coef','Score'})
for x,y in combinations(data.columns,2):
    data['%s_%s'%(x,y)] = data[x]/data[y]
    data['%s_%s' % (x, y)] = data['%s_%s'%(x,y)].replace(np.inf,np.nan)
    data['%s_%s' % (x, y)] = data['%s_%s'%(x,y)].replace(0,np.nan)
    tmp = data[['%s_%s'%(x,y), 'ccat']].copy()
    tmp = tmp.dropna(how='any',axis=0)
    x1 = tmp['%s_%s'%(x,y)].values.reshape(-1,1)
    ccat = tmp['ccat'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(x1, ccat, test_size=0.2, random_state=0)
    LR = linear_model.LinearRegression()
    LR.fit(X_train, y_train)
    rdata.loc['%s + %s'%(x,y),'Intercept'] = LR.intercept_
    rdata.loc['%s + %s'%(x,y),'Coef'] = LR.coef_
    rdata.loc['%s + %s'%(x,y),'Score'] = LR.score(X_test, y_test)
    plt.show()
    plt.scatter(x1,ccat,color='black')
    plt.title('%s + %s'%(x,y))
    plt.savefig('./figure/%s + %s.png'%(x,y))

rdata.to_excel('./regdata2.xlsx')

data.to_excel('./ccat.xlsx')

