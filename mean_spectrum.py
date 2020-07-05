import pandas as pd
import json
import numpy as np
import pickle
from sklearn import linear_model
from scipy.cluster.hierarchy import dendrogram


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



with open('/Users/siaga/Downloads/alkenone_bin.pkl','rb') as f:
    data = pickle.load(f)
data = data.T
data = data.dropna(subset=[557.25])

with open (r'./Dict/pixel_dict.json') as f:
    pixel_dict = json.load(f)
with open (r'./Dict/ccat_dict.json') as f:
    ccat_dict = json.load(f)

data['pixel'] = data.index.map(pixel_dict)
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
data=data.dropna(axis=1,thresh=0.8*data.shape[0])
del data['x']
del data['y']
data = data.drop(index=0)
data = data.replace(np.nan,0)
data.to_pickle(r'/Users/siaga/Downloads/alkenone_mean_bin.pkl')

