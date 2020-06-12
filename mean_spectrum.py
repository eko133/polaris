import pandas as pd
import json
import numpy as np
import main
import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet
import concurrent.futures
from itertools import combinations
from sklearn.decomposition import PCA


def simple_linear_regression(packed_arg):
    data, mass, outputtxt = packed_arg
    x, y = mass
    tmp = data[[x, y, 'ccat']].copy()
    tmp = tmp.dropna(how='any', axis=0)
    tmp['new'] = tmp[x]/(tmp[x]+tmp[y])
    tmp = tmp.dropna(how='any', axis=0)
    X_train, X_test, y_train, y_test = train_test_split(tmp['new'], tmp['ccat'], test_size=0.1, random_state=0)
    LR = linear_model.LinearRegression()
    LR.fit(np.array(X_train).reshape(-1, 1), y_train)
    score = LR.score(np.array(X_test).reshape(-1, 1), y_test)
    print(x,y,score)
    if score >= 0.3:
        with open(outputtxt, 'a') as f:
            f.writelines(f'{x, y}, {LR.coef_}, {LR.intercept_}, {score} \n')

# def ccat_label(ccat):
#     if ccat < 0.45:
#         return 1
#     elif ccat > 0.5:
#         return 2
#
#
with open('/Users/siaga/Downloads/SterolBin.pkl','rb') as f:
    data = pickle.load(f)
data = data.T
data = data.dropna(subset=[433.38])

with open (r'./Dict/pixel_dict.json') as f:
    pixel_dict = json.load(f)
with open (r'./Dict/ccat_dict.json') as f:
    ccat_dict = json.load(f)

data['pixel'] = data.index.map(pixel_dict)
data['ccat'] = data.index.map(ccat_dict)
data = data.dropna(subset=['ccat'])

data = data.dropna(subset=['pixel'])
data.loc[:, 'x'] = data.pixel.map(lambda x: x[0])
data.loc[:, 'y'] = data.pixel.map(lambda x: x[1])
del data['pixel']
data['x'] = 0.0418 * data['x']
data['y'] = 0.0418 * data['y']
start = data.y.min()
end = data.y.max()
spacing = int((end - start) / 0.2)
data['depth'] = 0
average_points = np.linspace(start, end, spacing, endpoint=True)
for i in range(len(average_points)-1):
    data.loc[(data['y'] >= average_points[i]) & (data['y'] < average_points[i + 1]),'depth'] = average_points[i]
data = data.groupby('depth').mean()
data=data.dropna(axis=1,thresh=1*data.shape[0])
del data['x']
del data['y']
data = data.drop(index=0)
data = data.replace(np.nan,0)
tmp = data.ccat
data = data.drop(columns=['ccat'])
data = data.T
for column in data.columns:
    data[column] = data[column]/data[column].sum()
data = data.T
data['ccat'] = tmp

with open('./SterolMean.pkl','wb') as f:
    pickle.dump(data,f)

pca = PCA(n_components=2)
pComponents = pca.fit_transform(data)
print(pca.explained_variance_ratio_)
pcaData = pd.DataFrame(data=pComponents, columns=['principal component 1', 'principal component 2'])
pcaData.index = data.index
loadings = pca.components_
loadings = loadings.T
loadings = pd.DataFrame(loadings)
loadings.index = data.T.index
loadings.to_csv(r'Data/sterol_loadings.csv')
pcaData.to_csv(r'Data/sterol_pca_results.csv')
# mass_lists_combinations =list(combinations(data.columns,2))
#
# args = ((data,mass,'./SharedStorage/shared.txt') for mass in mass_lists_combinations)
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     executor.map(simple_linear_regression, args)
#

# plt.imshow(data,cmap='hot',interpolation='nearest')
# plt.show()

# data['label'] = data['ccat'].apply(ccat_label)
# data = data.drop(columns=['ccat'])
# data = data.dropna(subset=['label'])

# data['label'] = data['label'].astype(str)
# X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='label'), data['label'], test_size=0.1)
# regr = svm.SVC()
# regr.fit(X_train,y_train)
# regr.score(X_test,y_test)
