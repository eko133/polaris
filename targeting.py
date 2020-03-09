# -*- coding: utf-8 -*-


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

dir = '/Users/siaga/Git/polaris/'

def filter(path, filter1):
    with open(path,'r') as f:
        lines = f.readlines()
    with open(r'%s_'%filter1+path,'w') as f:
        for line in lines:
            if filter1 in line:
                f.write(line)

def exclude_mass_list():
    gdgt0 = '1301.3154	1302.3187	1302.3227	1303.3260	1319.3492	1320.3526	651.6650	652.6683	668.6915	669.6949	660.1782	661.1816	1324.3046	1325.3080	1340.2785	1341.2819'
    gdgt5 = '1291.2371	1292.2405	1292.2444	1293.2478	1309.2710	1310.2743	646.6258	647.6292	663.6524	664.6558	655.1391	656.1425	1314.2264	1315.2297	1330.2003	1331.2037'
    exclude_mass_list = list()
    tmp_mass_list = gdgt0.split('\t') + gdgt5.split('\t')
    for i in range(len(tmp_mass_list)):
        mass = float(tmp_mass_list[i])
        mass = round(mass, 2)
        exclude_mass_list = exclude_mass_list + [str(mass), str(mass + 0.01), str(mass - 0.01)]
        with open(dir + r'dict/gdgt_excluded_mass_list.json', 'w') as f:
            json.dump(exclude_mass_list, f)

def delete_exclude_mass(path):
    with open(dir + r'dict/gdgt_excluded_mass_list.json', 'r') as f:
        exclude_mass_list = json.load(f)
    data = pd.read_csv(path)
    for column in data.columns:
        if column in exclude_mass_list:
            del data[column]
    data.to_csv(path)

def align(path):
    with open (path) as f:
        lines =f.readlines()
    samples = {}
    basket = pd.DataFrame()
    for line in lines:
        data = line.split(';')
        sample_name = data[0]
        samples[sample_name] = pd.DataFrame()
        del data[0]
        del data[0]
        data = pd.DataFrame(np.array(data).reshape((-1,3)),columns=['m/z',sample_name,'S/N'])
        data = data.drop(columns='S/N')
        data = data.astype(float)
        data['m/z'] = data['m/z'].round(2)
        data = data.groupby('m/z').agg({sample_name:sum})
        samples[sample_name] = data.copy()
    for sample in samples:
        basket = basket.merge(samples[sample], how ='outer', left_index=True, right_index=True)
    basket = basket.reset_index()

    duplicate_flag = 1
    while duplicate_flag == 1:
        mass_index = 0
        duplicate_flag = 0
        while mass_index < (len(basket)-1):
            if (basket.loc[mass_index+1,'m/z'] - basket.loc[mass_index,'m/z']) <= 0.01 and (basket.loc[mass_index+1,'m/z'] - basket.loc[mass_index,'m/z']) !=0 :
                duplicate_flag = 1
                basket.loc[mass_index,'m/z'] = basket.loc[mass_index+1,'m/z']
            mass_index = mass_index + 1
    basket = basket.groupby('m/z').sum()
    basket.to_csv(os.path.splitext(path)[0] + r'.csv')
    return basket

def normalizer(path):
    data = pd.read_csv(path)
    data = data.set_index('m/z')
    data = data.replace(0, np.nan)
    column_length = len(data.columns)
    data = data.dropna(thresh=0.2 * column_length)
    row_length = len(data)
    data = data.dropna(thresh=0.2 * row_length, axis=1)
    for column in data.columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    data = data.T
    for column in data.columns:
        if data[column].mean() >= 0.5:
            del data[column]
    data = data.T
    for column in data.columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    data = data.T
    data.to_csv(os.path.split(path)[0] + r'normalized_' + os.path.split(path)[1])

def pca(path):
    data = pd.read_csv(path)
    data = data.set_index('m/z')
    column_length = len(data.columns)
    data = data.dropna(thresh=0.5 * column_length)
    row_length = len(data)
    data = data.dropna(thresh=0.5 * row_length, axis=1)
    data = data.fillna(0)
    for column in data.columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    data = data.T
    for column in data.columns:
        if data[column].mean() >= 0.1:
            del data[column]
    data = data.T
    for column in data.columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    data = data.T

    pca = PCA(n_components=2)
    pComponents = pca.fit_transform(data)
    print(pca.explained_variance_ratio_)
    pcaData = pd.DataFrame(data=pComponents, columns=['principal component 1', 'principal component 2'])
    pcaData.index = data.index
    loadings = pca.components_
    loadings = loadings.T
    loadings = pd.DataFrame(loadings)
    loadings.index = data.T.index
    loadings.to_csv(dir+ r'data/sterol_loadings.csv')
    pcaData.to_csv(dir + r'data/sterol_pca_results.csv')

def get_important_compounds(path):
    loadings = pd.read_csv(path)
    loadings = loadings.abs()
    loadings = loadings[(loadings['0'] >= 0.1) | (loadings['1'] >=0.1)]
    important_compounds_list = list(loadings['m/z'])
    return important_compounds_list
    
def get_ccat_dict(path):
    data = pd.read_csv(path)
    data['x'] = data['x'].astype(str)
    data['x'] = data['x'].str.zfill(3)
    data['y'] = data['y'].astype(str)
    data['y'] = data['y'].str.zfill(3)
    data['sample'] = 'R00X' + data['x'] + 'Y' + data['y']
    ccat_dict = dict()
    ccat_dict = pd.Series(data['CCaT'].values,index = data['sample']).to_dict()
    
    data['combine_pixel'] = data[['pixel_x', 'pixel_y']].values.tolist()
    data['combine_pixel'] = tuple(data['combine_pixel'])
    pixel_dict = pd.Series(data['combine_pixel'].values,index = data['sample']).to_dict()

    with open (dir + r'dict/ccat_dict.json','w') as f:
        json.dump(ccat_dict,f)
    with open (dir + r'dict/pixel_dict.json','w') as f:
        json.dump(pixel_dict,f)

def linear_regression(path):
    data = pd.read_csv(path)

    data.drop(data.columns[0], axis=1, inplace=True)

    data = data.set_index(data.columns[0])
    # data = data.T
    #
    # important_compounds_list = get_important_compounds(r'./data/sterol_loadings.csv')
    
    # for column in data.columns:
    #     if column not in important_compounds_list:
    #         del data[column]
    
    # data['ccat'] =  data.index.map(ccat_dict)
    #
    # data = data.dropna(subset=['ccat'])
    
    rdata = pd.DataFrame(columns={'Intercept', 'Coef', 'Score'})
    
    # compounds = list(data.columns)
    # compounds.remove('ccat')
        
    for x, y in combinations(data.columns,2):
        ## not carbon isotopes and not carbon clusters
        difference = float(x) - float(y)
        difference_dem = math.modf(difference)[0]
        if (abs(difference) >= 2) & (abs(difference_dem)>0.01):
            tmp = data[[x,y]].copy()
            tmp = tmp.dropna(how='any', axis=0)
            if len(tmp) >= 50:
                print(x, y)
                x1 = tmp[x].values.reshape(-1, 1)
                y1 = tmp[y].values.reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=0)
                LR = linear_model.LinearRegression()
                LR.fit(X_train, y_train)
                rdata.loc['%s, %s' % (x, y), 'Intercept'] = LR.intercept_
                rdata.loc['%s, %s' % (x, y), 'Coef'] = LR.coef_
                rdata.loc['%s, %s' % (x, y), 'Score'] = LR.score(X_test, y_test)
        # plt.show()
        # plt.scatter(x1, ccat, color='black')
        # plt.title('%s + %s' % (x, y))
        # plt.savefig('./figure/%s + %s.png' % (x, y))
    rdata = rdata[rdata.Score >=0.2]

    rdata.to_csv(os.path.split(path)[0] + r'lr_' + os.path.split(path)[1])


def group_by_ccat():
    with open (dir+r'dict/ccat_dict.json', 'r') as f:
        ccat_dict = json.load(f)
    ccat_dict = pd.DataFrame.from_dict(ccat_dict,orient='index',columns=['ccat'])
    ccat_min = ccat_dict.ccat.min()
    ccat_max = ccat_dict.ccat.max()
    average_points = np.linspace(ccat_min, ccat_max, 111, endpoint=True)
    averaged_data = pd.DataFrame()
    group_by_ccat=dict()
    for i in range(len(average_points) - 2):
        data_tmp = ccat_dict[(ccat_dict.ccat >= average_points[i]) & (ccat_dict.ccat <= average_points[i + 1])]
        if len(data_tmp) >= 50:
            averaged_ccat = data_tmp.ccat.mean()
            grouped_sample = data_tmp.index.to_list()
            group_by_ccat[round(averaged_ccat,5)] = grouped_sample
    with open (dir + 'dict/group_by_ccat.json', 'w') as f:
        json.dump(group_by_ccat,f)


def target(path):
    ## find targeting compounds
    f = open(path,'r')
    lines =f.readlines()
    basket = pd.DataFrame()
    for line in lines:
        if 'R00' in line:
            data = line.split(';')
            sample_name = data[0]
            del data[0]
            del data[0]
            data = pd.DataFrame(np.array(data).reshape((-1,3)),columns=['m/z',sample_name,'S/N'])
            data = data.astype(float)
            data1 = data[(data['m/z'] >= 473.20) & (data['m/z'] <= 473.22)]
            data2 = data[(data['m/z'] >= 557.30) & (data['m/z'] <= 557.32)]
            compound1 = data1[sample_name].max()
            compound2 = data2[sample_name].max()
            basket.loc[sample_name,'newindices'] = compound1/(compound1+compound2)
    basket = basket.replace(1,np.nan)
    basket = basket.dropna()   
    basket['combine_pixel']= basket.index.map(pixel_dict)
    basket = basket.dropna()
    basket.loc[:,'pixel_x']=basket.combine_pixel.map(lambda x:x[0])
    basket.loc[:,'pixel_y']=basket.combine_pixel.map(lambda x:x[1])
    del basket['combine_pixel']
    ## converting pixel axis to actual axis in mm
    basket['pixel_x'] = 0.0418 * basket['pixel_x']
    basket['pixel_y'] = 0.0418 * basket['pixel_y']
    basket['ccat'] = basket.index.map(ccat_dict)
    basket.to_csv(r'pixel.csv')
    ## start averaging results
    start = basket.pixel_y.min()
    end = basket.pixel_y.max()
    average_points = np.linspace(start,end,227,endpoint=True)
    averaged_data = pd.DataFrame()
    for i in range(len(average_points)-2):
        data_tmp = basket[(basket['pixel_y'] >= average_points[i]) & (basket['pixel_y'] <= average_points[i+1])]
        if len(data_tmp)>= 10:
            ccat = data_tmp.ccat.mean()
            newindices = data_tmp.newindices.mean()
            y = data_tmp.pixel_y.mean()
            averaged_data.loc[y,'ccat'] = ccat
            averaged_data.loc[y,'newindices'] = newindices
    averaged_data.to_csv(r'averaged_data.csv')

def find_grouped_samples():
    with open (dir+ r'dict/group_by_ccat.json','r') as f:
        grouped_by_ccat = json.load(f)
    for key in grouped_by_ccat:
        grouped_sample =grouped_by_ccat[key]
        with open (dir+ r'data/gdgt_test.txt','r') as f:
            lines =f.readlines()
        with open (dir+r'data/grouped_by_ccat/ccat%s.txt'%key,'w') as f:
            for line in lines:
                data = line.split(';')
                if data[0] in grouped_sample:
                    f.writelines(line)



with open (dir+r'dict/ccat_dict.json','r')  as f:
    ccat_dict = json.load(f)

with open (dir+r'dict/pixel_dict.json','r')  as f:
    pixel_dict = json.load(f)

if __name__ == "__main__":
    main()
