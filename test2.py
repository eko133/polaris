<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# import pickle
# import pandas as pd
#
# f = open('./gdgt_similarMassMerged2.pkl','rb')
# samples = pickle.load(f)
# samples = samples.set_index('m/z')
# biomarker = pd.DataFrame(index=samples.columns)
# for column in samples:
#     gdgt_5 = samples.loc[1314.23,column]
#     gdgt_0 = samples.loc[1324.31,column]
#     biomarker.loc[column,'ccat1'] = gdgt_5/(gdgt_0+gdgt_5)
# biomarker = biomarker.dropna()
# biomarker.to_pickle('./ccat2.pkl')

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt

# def reg(x,y):
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#     LR = linear_model.LinearRegression()
#     LR.fit(X_train, y_train)
#     print('intercept_:%.3f' % LR.intercept_+'coef_:%.3f' % LR.coef_+'Variance score: %.3f' % r2_score(y_test, LR.predict(X_test))+'score: %.3f' % LR.score(X_test, y_test))
#
# data=pd.read_excel('/Users/siaga/Desktop/pca_processed.xlsx',index_col=0)
data = pd.read_pickle('./gdgt_similarMassMerged_Y50.pkl')
data = data.set_index('m/z')
column_length = len(data.columns)
data = data.dropna(thresh=0.5*column_length)
row_length = len(data)
data = data.dropna(thresh=0.5*row_length, axis=1)
data = data.fillna(0)
# data = (data - data.min())/(data.max()-data.min())
data = data.T
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
# data=data.T
# pca=PCA(n_components=2)
# pComponents=pca.fit_transform(data)
# print(pca.explained_variance_ratio_)
# pcaData=pd.DataFrame(data=pComponents,columns = ['principal component 1', 'principal component 2'])
# pcaData.index=data.index
# loadings=pca.components_
# loadings=loadings.T
# loadings=pd.DataFrame(loadings)
# loadings.index=data.T.index
# loadings.to_excel('/Users/siaga/Desktop/loadings.xlsx')
# pcaData.to_excel('/Users/siaga/Desktop/pca_results.xlsx')
=======
import pickle
f = open('./samples_line_test.p','rb')
samples = pickle.load(f)
>>>>>>> 2cfc163034ecf24df37ccc7cad1f3fb1ed7182fb
=======
import pickle
f = open('./samples_line_test.p','rb')
samples = pickle.load(f)
>>>>>>> 2cfc163034ecf24df37ccc7cad1f3fb1ed7182fb
=======
import pickle
f = open('./samples_line_test.p','rb')
samples = pickle.load(f)
>>>>>>> 2cfc163034ecf24df37ccc7cad1f3fb1ed7182fb
