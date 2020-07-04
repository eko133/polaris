import numpy as np
import pandas as pd
from sklearn import linear_model
import platform
import os


if platform.system() == 'Darwin':
    shared_storage = '/Users/siaga/Dropbox/SharedStorage/polaris'


def mp_mass_bining(packed_args):
    """
    Mass peak bining function with the interval of 0.01 Da, suitable for multiprocessing.
    """
    lines, datan = packed_args
    for line in lines:
        data = line.split(';')
        sample_name = data[0]
        print(sample_name)
        del data[0]
        del data[0]
        tmp = pd.DataFrame(np.array(data).reshape((-1, 3)), columns=['m/z', sample_name, 'S/N'])
        tmp = tmp.drop(columns='S/N')
        tmp = tmp.astype(float)
        tmp.loc[tmp[sample_name] >= tmp[sample_name].quantile(0.95), sample_name] = 0
        tmp = tmp.dropna()
        tmp['m/z'] = tmp['m/z'].round(2)
        tmp = tmp.groupby('m/z').max()
        datan = datan.merge(tmp, how='outer', left_index=True, right_index=True)
    return datan


def peak_normalize(raw_data, *args, normalize='none'):
    """
    Normalize peak intensities in different laser points for better cross-comparison
    None = Just drop the outliers and leave the peak intensities untouched
    tic = Drop the outliers and then normalize peak intensities to TIC
    median = Drop the outliers and then normalize peak intensities according to median values
    vectornorm = np.sqrt((peak intensity**2).sum())
    """
    if args:
        tmp_columns = args
        tmp = pd.DataFrame(raw_data[tmp_columns], columns=tmp_columns)
        raw_data = raw_data.drop(columns=tmp_columns)
    data = raw_data.T
    for column in data.columns:
        data.loc[data[column] > data[column].replace(0,np.nan).dropna().quantile(0.98), column] = 0
        if normalize == 'none':
            pass
        elif normalize == 'tic':
            data[column] = data[column] / data[column].sum()
        elif normalize == 'median':
            data[column] = data[column] / data[column].replace(0,np.nan).dropna().median()
        elif normalize == 'vector':
            data[column] = data[column] / np.sqrt((data[column] ** 2).sum())
    if args:
        data[tmp_columns] = tmp
    return data.T


def mp_colinear_finder(packed_arg):
    """
    Find peaks show colinearity in varying laser points, possible isotope peaks or peaks with different adducts
    """
    data, mass = packed_arg
    x, y = mass
    tmp = data[[x, y]].copy()
    tmp = tmp.dropna(how='any', axis=0)
    if len(tmp)>=100:
        print(mass)
        LR = linear_model.LinearRegression()
        LR.fit(np.array(tmp[x]).reshape(-1,1), tmp[y])
        score = LR.score(np.array(tmp[x]).reshape(-1,1), tmp[y])
        with open(os.path.join(shared_storage,'alkenone_colinear.txt'),'a') as f:
            f.writelines(f'{x, y,score} \n')


