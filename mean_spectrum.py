import pandas as pd
import json
import numpy as np
import main
import pickle

with open('./SterolBin.pkl','rb') as f:
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
data=data.dropna(axis=1,thresh=0.8*data.shape[0])
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
