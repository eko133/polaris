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
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
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

# pca = pd.DataFrame()
# with open (r'./pkl/negative_ESI_result.pkl','rb') as f:
#     data = pickle.load(f)
# for i in data:
#     data[i] = data[i][(data[i]['Class']!='') & (data[i]['Class']!='O4') & (data[i]['Class']!='O4N1')& (data[i]['Class']!='O3') ]
#     data[i] = data[i].dropna()
#     data[i] = data[i][data[i]['Class']=='N1']
#     data[i]['dbe'] = data[i]['dbe'].astype(int)
#     # data[i] = data[i].drop(data[i][(data[i].dbe == 1) & (data[i].C == 16)].index)
#     # data[i] = data[i].drop(data[i][(data[i].dbe == 1) & (data[i].C == 18)].index)
#     # data[i] = data[i].drop(data[i][(data[i].dbe == 2) & (data[i].C == 18)].index)
#     #
#     data[i] = data[i][(data[i]['dbe']>=9) &(data[i]['dbe']<=15) ]
#     data[i] = data[i][['em','I','Class']]
#     data[i].em = data[i].em.round(6)
#
#     # data[i].em = data[i].em.astype(str)
#     # data[i].em = data[i].em + ','+ data[i].Class
#     del data[i]['Class']
#     data[i]['I'] = (data[i]['I'] - data[i]['I'].min())/(data[i]['I'].max() - data[i]['I'].min())
#     data[i] = data[i].rename(columns={'I':i})
#     data[i] = data[i].set_index('em')
#     pca = pca.merge(data[i],how='outer',left_index=True,right_index=True)
# pca=pca.T
# # pca= pca.dropna(axis=1)
# pca = pca.replace(np.nan,0)

pca = pd.read_clipboard()


sample_list = pca.index
clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(pca)
plot_dendrogram(clustering,labels = pca.index)
plt.savefig('plt1.png', dpi=600, format='png', bbox_inches='tight')
plt.show()