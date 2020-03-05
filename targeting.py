import pandas as pd
import numpy as np

## find targeting compounds
f = open(r'/Users/siaga/gdgt_test.txt','r')
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
basket.to_excel('./target1345_all.xlsx')
data = basket.copy()

## Averaging indices on horizontal scale
data=pd.read_excel(r'./target1345_all.xlsx')
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
