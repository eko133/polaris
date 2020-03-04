import pandas as pd

data=pd.read_excel('./target1345_all.xlsx')
data['sample'] = data['sample'].str.replace('R00X','')
data['X'] = data['sample'].str.split('Y').str[0].astype(int)
data['Y'] = data['sample'].str.split('Y').str[1].astype(int)
del data['sample']
data['combine_axis'] = data[['X', 'Y']].values.tolist()
data['combine_axis'] = tuple(data['combine_axis'])


## generating dictionary
dict_orig = pd.read_excel('~/Desktop/SBB_0-5_CCaTxy_transformed.xlsx')
dict_orig['combine_axis'] = dict_orig[['x', 'y']].values.tolist()
dict_orig['combine_axis'] = tuple(dict_orig['combine_axis'])

dict_orig['combine_pixel'] = dict_orig[['x.1', 'y.1']].values.tolist()
dict_orig['combine_pixel'] = tuple(dict_orig['combine_pixel'])
dict = pd.Series(dict_orig.combine_pixel.values,index=dict_orig.combine_axis).to_dict()


data['combine_pixel']= data['combine_axis'].map(dict)
# data.to_excel('./pixel.xlsx')
data = data.dropna()

data.loc[:,'pixel_x']=data.combine_pixel.map(lambda x:x[0])
data.loc[:,'pixel_y']=data.combine_pixel.map(lambda x:x[1])

data['pixel_x'] = 0.0418 * data['pixel_x']
data['pixel_y'] = 0.0418 * data['pixel_y']

data.to_excel('./pixel.xlsx')

