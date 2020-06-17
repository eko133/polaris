import pandas as pd
import json
import numpy as np
import main
import pickle
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet
import concurrent.futures
from itertools import combinations
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def simple_linear_regression(packed_arg):
    data, mass, outputtxt = packed_arg
    x, y = mass
    tmp = data[[x, y, 'ccat']].copy()
    tmp = tmp.dropna(how='any', axis=0)
    tmp['new'] = tmp[x]/(tmp[x]+tmp[y])
    tmp = tmp.dropna(how='any', axis=0)
    LR = linear_model.LinearRegression()
    LR.fit(np.array(tmp['new']).reshape(-1, 1), tmp['ccat'])
    score = LR.score(np.array(tmp['new']).reshape(-1, 1), tmp['ccat'])
    print(x,y,score)
    with open(outputtxt, 'a') as f:
        f.writelines(f'{x, y}, {LR.coef_}, {LR.intercept_}, {score} \n')

def colinear_finder(packed_arg):
    data, mass = packed_arg
    x, y = mass
    tmp = data[[x, y]].copy()
    tmp = tmp.dropna(how='any', axis=0)
    LR = linear_model.LinearRegression()
    LR.fit(np.array(tmp[x]).reshape(-1,1), tmp[y])
    score = LR.score(np.array(tmp[x]).reshape(-1,1), tmp[y])
    print(x,y,score)
    with open('./SharedStorage/mean_colinear.txt', 'a') as f:
        f.writelines(f'{x, y,score} \n')


def ccat_label(ccat):
    if ccat < 0.48:
        return 1
    elif (ccat >= 0.48) & (ccat<0.5):
        return 2
    elif ccat>=0.5:
        return 3


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
data['depth']=data['depth'].round(1)
data = data.groupby('depth').mean()
data=data.dropna(axis=1,thresh=1*data.shape[0])
del data['x']
del data['y']
data = data.drop(index=0)
data = data.replace(np.nan,0)
##exclude colinear m/z
colinear = pd.read_csv('./SharedStorage/mean_colinear.csv')
colinear = colinear[colinear['score']>=0.7]
colinear_list = colinear['y'].tolist()
for mass in tuple(colinear_list):
    try:
        data = data.drop(columns=mass)
    except KeyError:
        continue

tmp = pd.DataFrame(data.ccat,columns=['ccat'])
data = data.drop(columns=['ccat',539.24])

# #median normalization
# data['median'] = data.median(axis=1)
# for row in data.index:
#     scal = data.loc[row,'median']
#     data.loc[row,:] = data.loc[row,:]/scal
# data=data.drop(columns=['median'])

# data = data.T
# for column in data.columns:
#     data[column] = data[column]/data[column].sum()
# data = data.T
# data['ccat'] = tmp
# #
# # with open('./SterolMean.pkl','wb') as f:
# #     pickle.dump(data,f)
# #
# # data['label'] = data['ccat'].apply(ccat_label)
# # data = data.drop(columns=['ccat'])
# # data = data.dropna(subset=['label'])
# #
# # data['label'] = data['label'].astype(str)
# #
# # clf = linear_model.LogisticRegressionCV()
# # clf.fit(data.drop(columns='label'), data['label'])
for column in data.columns:
    median = data[column][data[column].between(data[column].quantile(0.01),data[column].quantile(0.99))].median()
    data.loc[data[column]>data[column].quantile(0.99),column] = median
    data.loc[data[column]<data[column].quantile(0.01),column] = median

# # median normalization
# data['median'] = data.median(axis=1)
# for row in data.index:
#     scal = data.loc[row,'median']
#     data.loc[row,:] = data.loc[row,:]/scal
# data=data.drop(columns=['median'])

# for i in [(392.15,517.27),(392.15,460.25),(393.3,433.26),(405.19,517.27)]:
#     data[f'{i[0]}/{i[1]}'] = data[i[0]]/(data[i[0]]+data[i[1]])
#     plt.figure()
#     sns.scatterplot(data[f'{i[0]}/{i[1]}'], tmp['ccat'])
#     plt.savefig(f'{i[0]}+{i[1]}.png')
#     print(stats.spearmanr(data[f'{i[0]}/{i[1]}'],tmp['ccat']))

scaled_features = StandardScaler().fit_transform(data.values)
scaled_data = pd.DataFrame(scaled_features,index=data.index,columns=data.columns)

pca = PCA(n_components=10)
pComponents = pca.fit_transform(scaled_data)
print(pca.explained_variance_ratio_)
pcaData = pd.DataFrame(data=pComponents, columns=[[f'principal component {i}' for i in range(1,11)]])
pcaData.index = data.index
loadings = pca.components_
loadings = loadings.T
loadings = pd.DataFrame(loadings)
loadings.index = data.T.index
loadings.to_csv(r'Data/sterol_loadings.csv')
pcaData.to_csv(r'Data/sterol_pca_results.csv')
#
# mass_lists_combinations =list(combinations(data.columns,2))
#
# args = ((data,mass) for mass in mass_lists_combinations)
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     executor.map(colinear_finder, args)
#

# plt.imshow(data,cmap='hot',interpolation='nearest')
# plt.show()

# lasso = linear_model.LassoCV()
# lasso.fit(data.drop(columns='ccat'),data['ccat'])

# tmp['label'] = tmp['ccat'].apply(ccat_label)
# # # data = data.drop(columns=['ccat'])
# # # data = data.dropna(subset=['label'])
# # #
#
# data = data.replace(np.nan,0)
# tmp = tmp.replace(np.nan,0)
# tmp['label'] = tmp['label'].astype(str)
# X_train, X_test, y_train, y_test = train_test_split(data, tmp['label'], test_size=0.1)
# regr = svm.SVC()
# regr.fit(X_train,y_train)
# regr.score(X_test,y_test)
# #
# # Z = hierarchy.linkage(scaled_data)
# hierarchy.dendrogram(Z,labels=scaled_data.index)