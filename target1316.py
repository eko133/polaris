import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt
import sys
import json
import os
import math

# set target m/z
target1 = 413.14
target2 = 479.18
target_raw_txt = '/Users/siaga/Documents/gdgt/sbb_sterol.txt'

with open (r'./dict/ccat_dict.json','r')  as f:
    ccat_dict = json.load(f)

with open (r'./dict/pixel_dict.json','r')  as f:
    pixel_dict = json.load(f)

with open (target_raw_txt) as f:
    lines = f.readlines()
basket = pd.DataFrame()
for line in lines:
    if 'R00' in line:
        data = line.split(';')
        sample_name = data[0]
        del data[0]
        del data[0]
        data = pd.DataFrame(np.array(data).reshape((-1, 3)), columns=['m/z', sample_name, 'S/N'])
        data = data.astype(float)
        data1 = data[(data['m/z'] >= (target1-0.1)) & (data['m/z'] <= (target1+0.1))]
        data2 = data[(data['m/z'] >= (target2-0.1)) & (data['m/z'] <= (target2+0.1))]
        compound1 = data1[sample_name].max()
        compound2 = data2[sample_name].max()
        basket.loc[sample_name, 'newindices'] = compound1 / (compound1 + compound2)
basket = basket.replace(1, np.nan)
basket = basket.dropna()
basket['combine_pixel'] = basket.index.map(pixel_dict)
basket = basket.dropna()
basket.loc[:, 'pixel_x'] = basket.combine_pixel.map(lambda x: x[0])
basket.loc[:, 'pixel_y'] = basket.combine_pixel.map(lambda x: x[1])
del basket['combine_pixel']
## converting pixel axis to actual axis in mm
basket['pixel_x'] = 0.0418 * basket['pixel_x']
basket['pixel_y'] = 0.0418 * basket['pixel_y']
basket['ccat'] = basket.index.map(ccat_dict)
basket.to_csv(r'./pixel.csv')
## start averaging results
start = basket.pixel_y.min()
end = basket.pixel_y.max()
average_points = np.linspace(start, end, 227, endpoint=True)
averaged_data = pd.DataFrame()
for i in range(len(average_points) - 2):
    data_tmp = basket[(basket['pixel_y'] >= average_points[i]) & (basket['pixel_y'] <= average_points[i + 1])]
    if len(data_tmp) >= 10:
        ccat = data_tmp.ccat.mean()
        newindices = data_tmp.newindices.mean()
        y = data_tmp.pixel_y.mean()
        averaged_data.loc[y, 'ccat'] = ccat
        averaged_data.loc[y, 'newindices'] = newindices
averaged_data.to_csv(r'./averaged_data.csv')
