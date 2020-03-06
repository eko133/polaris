import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt
import sys

def filter(path, filter1):
    with open(path,'r') as f:
        lines = f.readlines()
    with open(r'%s_'%filter1+path,'w') as f:
        for line in lines:
            if filter1 in line:
                f.write(line)

def align(path):
    f = open(path,'r')
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
    basket.to_csv(r'./data/SterolSimilarMassMerged.csv')
    return basket

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
        if data[column].mean() >= 0.5:
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
    loadings.to_excel(r'./data/sterol_loadings.xlsx')
    pcaData.to_excel(r'./data/sterol_pca_results.xlsx')

def LR(path):
    data = pd.read_pickle(path)
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
        if data[column].mean() >= 0.5:
            del data[column]
    data = data.T
    for column in data.columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

    data['ccat'] = data[1314.23] / (data[1314.23] + data[1324.31])
    data['ccat'] = data['ccat'].replace(1, np.nan)
    data['ccat'] = data['ccat'].replace(0, np.nan)
    data = data.dropna(how='any', axis=0)
    rdata = pd.DataFrame(columns={'Intercept', 'Coef', 'Score'})
    for x, y in combinations(data.columns, 2):
        data['%s_%s' % (x, y)] = data[x] / data[y]
        data['%s_%s' % (x, y)] = data['%s_%s' % (x, y)].replace(np.inf, np.nan)
        data['%s_%s' % (x, y)] = data['%s_%s' % (x, y)].replace(0, np.nan)
        tmp = data[['%s_%s' % (x, y), 'ccat']].copy()
        tmp = tmp.dropna(how='any', axis=0)
        x1 = tmp['%s_%s' % (x, y)].values.reshape(-1, 1)
        ccat = tmp['ccat'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(x1, ccat, test_size=0.2, random_state=0)
        LR = linear_model.LinearRegression()
        LR.fit(X_train, y_train)
        rdata.loc['%s + %s' % (x, y), 'Intercept'] = LR.intercept_
        rdata.loc['%s + %s' % (x, y), 'Coef'] = LR.coef_
        rdata.loc['%s + %s' % (x, y), 'Score'] = LR.score(X_test, y_test)
        plt.show()
        plt.scatter(x1, ccat, color='black')
        plt.title('%s + %s' % (x, y))
        plt.savefig('./figure/%s + %s.png' % (x, y))

    rdata.to_excel(r'./data/regdata2.xlsx')

    data.to_excel(r'./data/ccat.xlsx')

def target(path):
    ## find targeting compounds
    f = open(path,'r')
    lines =f.readlines()
    basket = pd.DataFrame(columns=['ccat','ccat_new'])
    for line in lines:
        data = line.split(';')
        sample_name = data[0]
        del data[0]
        del data[0]
        data = pd.DataFrame(np.array(data).reshape((-1,3)),columns=['m/z',sample_name,'S/N'])
        data = data.astype(float)
        data1 = data[(data['m/z'] >= 1345.00) & (data['m/z'] <= 1345.02)]
        data3 = data[(data['m/z'] >= 1324.29) & (data['m/z'] <= 1324.31)]
        data4 = data[(data['m/z'] >= 1314.220) & (data['m/z'] <= 1314.230)]
        t1314 = data4[sample_name].max()
        t1324 = data3[sample_name].max()
        t1345 = data1[sample_name].max()
        basket.loc[sample_name,'ccat'] = t1314/(t1314+t1324)
        basket.loc[sample_name,'ccat_new'] = t1314/(t1345+t1314)
    basket = basket.replace(1,np.nan)
    basket = basket.dropna()
    basket.to_excel('./data/target1345_all.xlsx')
    data = basket.copy()

    ## Averaging indices on horizontal scale
    data=pd.read_excel(r'./data/target1345_all.xlsx')
    data['sample'] = data['sample'].str.replace('R00X','')
    data['X'] = data['sample'].str.split('Y').str[0].astype(int)
    data['Y'] = data['sample'].str.split('Y').str[1].astype(int)
    del data['sample']
    data['combine_axis'] = data[['X', 'Y']].values.tolist()
    data['combine_axis'] = tuple(data['combine_axis'])


    ## generating dictionary (x,y) axis to pixel axis
    dict_orig = pd.read_excel('~/Desktop/SBB_0-5_CCaTxy_transformed.xlsx')
    dict_orig['combine_axis'] = dict_orig[['x', 'y']].values.tolist()
    dict_orig['combine_axis'] = tuple(dict_orig['combine_axis'])

    dict_orig['combine_pixel'] = dict_orig[['x.1', 'y.1']].values.tolist()
    dict_orig['combine_pixel'] = tuple(dict_orig['combine_pixel'])
    dict = pd.Series(dict_orig.combine_pixel.values,index=dict_orig.combine_axis).to_dict()


    data['combine_pixel']= data['combine_axis'].map(dict)
    data = data.dropna()

    data.loc[:,'pixel_x']=data.combine_pixel.map(lambda x:x[0])
    data.loc[:,'pixel_y']=data.combine_pixel.map(lambda x:x[1])

    ## converting pixel axis to actual axis in mm
    data['pixel_x'] = 0.0418 * data['pixel_x']
    data['pixel_y'] = 0.0418 * data['pixel_y']

    data.to_excel(r'./pixel.xlsx')

    ## start averaging results
    start = data.pixel_y.min()
    end = data.pixel_y.max()
    average_points = np.linspace(start,end,227,endpoint=True)
    averaged_data = pd.DataFrame()
    for i in range(len(average_points)-2):
        data_tmp = data[(data['pixel_y'] >= average_points[i]) & (data['pixel_y'] <= average_points[i+1])]
        if len(data_tmp)>= 10:
            ccat = data_tmp.ccat.mean()
            ccat_new = data_tmp.ccat_new.mean()
            y = data_tmp.pixel_y.mean()
            averaged_data.loc[y,'ccat'] = ccat
            averaged_data.loc[y,'ccat_new'] = ccat_new

    averaged_data.to_excel(r'./averaged_data.xlsx')

option = sys.argv[1]
path = sys.argv[2]

if option == 'a':
    print('align rawdata')
    align(path)

if option == 'p':
    print('pca analysis')
    pca(path)

if option == 'l':
    print('linear regression analysis')
    LR(path)

if option == 't':
    print('targeting specific compounds')
    target(path)

if 'filter' in option:
    option = option.strip('filter')
    print('filtering by %s'%option)
    filter(path,option)
