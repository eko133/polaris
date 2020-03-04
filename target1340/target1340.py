import os
import pandas as pd
import numpy as np

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

