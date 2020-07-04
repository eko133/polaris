import pandas as pd
import main
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab

params = {'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

def custom_heatmap(data):
    plt.xticks(np.arange(375,520,10))
    plt.yticks(np.arange(2,42,5))
    sns.heatmap(data)
    plt.show()

def custom_scatterplot(data):
    plt.figure(figsize=(6,10))
    plt.xlim()
    plt.ylim(2,42)
    sns.scatterplot(data)
    plt.show()

# mean spectrum
data = pd.read_pickle(r'~/Downloads/alkenone_mean_bin.pkl')
# colinearity info
colinear = pd.read_pickle(r'./SharedStorage/alkenone_mean_clinear.pkl')


##exclude colinear m/z
colinear = colinear[colinear['score']>=0.3]
colinear_list = colinear['y'].tolist()
dropped = 0
for mass in tuple(colinear_list):
    try:
        data = data.drop(columns=mass)
        dropped += 1
    except KeyError:
        continue

## normalize
data = main.peak_normalize(data,normalize='vector')
pca_data, loadings =main.custom_pca(data)
plt.figure(figsize=(6, 10))
plt.ylim(45, 0)
sns.scatterplot(pca_data['PC1'],pca_data.index)
plt.show()



#
# # #
# # # with open('./SterolMean.pkl','wb') as f:
# # #     pickle.dump(data,f)
# # #
# # # data['label'] = data['ccat'].apply(ccat_label)
# # # data = data.drop(columns=['ccat'])
# # # data = data.dropna(subset=['label'])
# # #
# # # data['label'] = data['label'].astype(str)
# # #
# # # clf = linear_model.LogisticRegressionCV()
# # # clf.fit(data.drop(columns='label'), data['label'])
#
#
#

# #
# # mass_lists_combinations =list(combinations(data.columns,2))
# #
# # args = ((data,mass) for mass in mass_lists_combinations)
# # with concurrent.futures.ProcessPoolExecutor() as executor:
# #     executor.map(colinear_finder, args)
# #
#
# # plt.imshow(data,cmap='hot',interpolation='nearest')
# # plt.show()
#
# # lasso = linear_model.LassoCV()
# # lasso.fit(data.drop(columns='ccat'),data['ccat'])
#
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
# # #
# # # Z = hierarchy.linkage(scaled_data)
# # hierarchy.dendrogram(Z,labels=scaled_data.index)