import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from util.preprocessing import peak_normalize
from util.decomposition import factorize
import platform
import os
from util.plotting import extract_coordinates
from util.plotting import coordinates_transform
from util.plotting import image_warping
from util.plotting import enhance_contrast
import pickle


if platform.system() == 'Darwin':
    shared_storage = '/Users/siaga/Dropbox/SharedStorage/polaris'


def color(bin_data,xray):
    x = bin_data.pixel_x
    y = bin_data.pixel_y
    return xray.loc[(xray.x==x)&(xray.y==y),'z']


with open(r'../Dict/affine_trans_dict.pkl','rb') as f:
    affine_trans_dict = pickle.load(f)


def trans(x,y):
    pixel = np.dot(affine_trans_dict['uk']['A'],[x,y])+affine_trans_dict['uk']['t']
    return pixel


bin_data = pd.read_pickle(os.path.join(shared_storage,'alkenone_bin.pkl'))
bin_data = bin_data.T
bin_data = bin_data.dropna(axis=1,thresh=0.1*bin_data.shape[0])
bin_data = peak_normalize(bin_data,normalize='tic')
bin_data = bin_data.replace(np.nan,0)
pca_data, loadings = factorize(bin_data,method='ica',n_components=10)
bin_data['x'],bin_data['y'] = extract_coordinates(bin_data)
pca_data['x'],pca_data['y'] = extract_coordinates(bin_data)
pc=dict()
for i in range(1,11):
    pc[i] = pca_data[['x','y',f'F{i}']]
    plt.imshow(pc[i].pivot('x','y',f'F{i}'),cmap='gray')
    plt.show()

# bin_data['xy'] = bin_data.index
# bin_data['x'] = bin_data.xy.str.extract(r'R00X(.*?)Y').astype(int)
# bin_data['y'] = bin_data.xy.str.extract(r'Y(.*?)$').astype(int)
#
# for i in range(1,21):
#     mass = loadings[f'PC{i}'].idxmax()
#     plt.imshow(bin_data.pivot('x','y',mass))
#     plt.show()

# bin_data['x_warped'] = image_warping(bin_data,0.4)
xray = pd.read_pickle(os.path.join(shared_storage,'X-Ray_pixel.pkl'))
bin_data['pixel'] = bin_data.apply(lambda m: trans(m.x, m.y), axis=1)
bin_data[['pixel_x','pixel_y']] = pd.DataFrame(bin_data.pixel.tolist(), index= bin_data.index)

with open(os.path.join(shared_storage,'pixel_dict.pkl'),'rb') as f:
    pixel_dict = pickle.load(f)

bin_data['pixel_xy'] = bin_data.apply(lambda x: f'{int(x.pixel_x)},{int(x.pixel_y)}',axis=1)
bin_data['color'] = bin_data.pixel_xy.map(pixel_dict['color'])
bin_data['age'] = bin_data.pixel_xy.map(pixel_dict['age'])


# def mass_correction(x, coef, intercept, mass_min, mass_max):
#     if x >= mass_min and x <= mass_max:
#         err = coef * x + intercept
#         return x / (1 + err / 1000000)
#     else:
#         return x
#
# test_txt = r'/Users/siaga/Dropbox/Documents/MALDI/SBB0-5_alkenone.txt'
# mass_calibration_list = [553.5319,551.5162,567.5475,559.4274,573.4067,569.4540,551.4435,557.2523,555.2367]
# compound = main.generate_possible_formula(520,580)
# with open(test_txt) as f:
#     lines = f.readlines()
#
# mass_calibration_list.sort()
# line = lines[1]
# data = line.split(';')
# sample_name = data[0]
# print(sample_name)
# del data[0]
# del data[0]
# tmp = pd.DataFrame(np.array(data).reshape((-1, 3)), columns=['m/z', 'I', 'S/N'])
# tmp = tmp.drop(columns='S/N')
# tmp = tmp.astype(float)
# tmp_cali = tmp.copy()
#
# for key in compound:
#     key_min, key_max = key - 0.0025, key + 0.0025
#     tmp1 = tmp[(tmp['m/z'] >= key_min) & (tmp['m/z'] <= key_max)]
#     try:
#         id = tmp1.idxmax()[1]
#         tmp.loc[id, 'em'] = key
#     except ValueError:
#         continue
# tmp = tmp.dropna()
# tmp.loc[:, 'ppm'] = 1000000 * (tmp.loc[:, 'm/z'] - tmp.loc[:, 'em']) / tmp.loc[:, 'em']
# tmp.loc[:, 'em'] = tmp.loc[:, 'em'].round(4)
# tmp = tmp.dropna(0)
# print(len(tmp))
# print(len(tmp.loc[tmp['m/z'].between(551,570),'ppm']))
#
# ax = plt.subplot(111)
# plt.xlim(-10,10)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.set_xlabel('mass error (ppm)', fontsize=20)
# ax.set_ylabel('frequency', fontsize=20)
# # sns.distplot(tmp.loc[tmp['m/z'].between(551,570),'ppm'],bins=20)
# sns.distplot(tmp['ppm'],bins=20)
#
#
# for mass in mass_calibration_list:
#     mass_min, mass_max = mass - 0.0025, mass + 0.0025
#     tmp1 = tmp[(tmp['m/z'] >= mass_min) & (tmp['m/z'] <= mass_max)]
#     try:
#         idx = tmp1.idxmax()[1]
#         tmp_cali.loc[idx, 'em'] = mass
#     except ValueError:
#         continue
# tmp_cali = tmp_cali.dropna()
# tmp_cali.loc[:, 'ppm'] = 1000000 * (tmp_cali.loc[:, 'm/z'] - tmp_cali.loc[:, 'em']) / tmp_cali.loc[:, 'em']
# tmp_cali.loc[:, 'em'] = tmp_cali.loc[:, 'em'].round(4)
# cali = dict(zip(tmp_cali['m/z'], tmp_cali['ppm']))
#
#
# mass_calibration_list = list(cali.keys())
# mass_calibration_list.sort()
# for i in range(len(mass_calibration_list) - 1):
#     coef = (cali[mass_calibration_list[i + 1]] - cali[mass_calibration_list[i]]) / (
#             mass_calibration_list[i + 1] - mass_calibration_list[i])
#     intercept = cali[mass_calibration_list[i]] - coef * mass_calibration_list[i]
#     if i == 0:
#         tmp['m/z'] = tmp['m/z'].apply(
#             lambda x: mass_correction(x, 0, cali[mass_calibration_list[i]], 0, mass_calibration_list[i]))
#     elif i == (len(mass_calibration_list) - 2):
#         tmp['m/z'] = tmp['m/z'].apply(
#             lambda x: mass_correction(x, 0, cali[mass_calibration_list[i + 1]], mass_calibration_list[i + 1],
#                                       1000))
#     tmp['m/z'] = tmp['m/z'].apply(
#         lambda x: mass_correction(x, coef, intercept, mass_calibration_list[i], mass_calibration_list[i + 1]))
# for key in compound:
#     key_min, key_max = key - 0.0025, key + 0.0025
#     tmp1 = tmp[(tmp['m/z'] >= key_min) & (tmp['m/z'] <= key_max)]
#     try:
#         id = tmp1.idxmax()[1]
#         tmp.loc[id, 'em'] = key
#     except ValueError:
#         continue
# tmp = tmp.dropna()
# tmp.loc[:, 'ppm'] = 1000000 * (tmp.loc[:, 'm/z'] - tmp.loc[:, 'em']) / tmp.loc[:, 'em']
# tmp.loc[:, 'em'] = tmp.loc[:, 'em'].round(4)
# sns.distplot(tmp['ppm'],bins=20)
# plt.show()
# tmp = tmp.dropna(0)
# print(len(tmp))
# print(len(tmp.loc[tmp['m/z'].between(551,570),'ppm']))

# show_im(bin_data)
# #
# X = bin_data.drop(columns = {'xy','x','y','pixel_x','pixel_y','pixel_xy','color'})
# X = pca_data.filter(regex='PC')
# y = bin_data['color']
# ld=LinearDiscriminantAnalysis()
# ld.fit(X,y)
# ld.score(X,y)

# X=main.peak_normalize(X,normalize='tic')
# X=X.replace(np.nan,0)
#
#
#
