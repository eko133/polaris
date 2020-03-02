import os
import pandas as pd
import numpy as np

f = open(r'/Users/siaga/y53_test.txt','r')
lines =f.readlines()
basket = pd.DataFrame(columns=['ccat','ccat_new'])
for line in lines:
    data = line.split(';')
    sample_name = data[0]
    del data[0]
    del data[0]
    data = pd.DataFrame(np.array(data).reshape((-1,3)),columns=['m/z',sample_name,'S/N'])
    data = data.astype(float)
    data1 = data[(data['m/z'] >= 1330.18) & (data['m/z'] <= 1330.20)]
    data2 = data[(data['m/z'] >= 1340.26) & (data['m/z'] <= 1340.28)]
    data3 = data[(data['m/z'] >= 1324.29) & (data['m/z'] <= 1324.31)]
    data4 = data[(data['m/z'] >= 1314.220) & (data['m/z'] <= 1314.230)]
    t1314 = data4[sample_name].max()
    t1324 = data3[sample_name].max()
    t1330 = data1[sample_name].max()
    t1340 = data2[sample_name].max()
    ccat = t1324/(t1314+t1324)
    ccat_new = t1340/(t1340+t1330)
    basket.loc[sample_name,'ccat'] = t1314/(t1314+t1324)
    basket.loc[sample_name,'ccat_new'] = t1330/(t1340+t1330)
basket = basket.replace(1,np.nan)
basket = basket.dropna()
basket.to_excel('./target1340.xlsx')

