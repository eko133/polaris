import pandas as pd
import os

data=pd.DataFrame()
os.chdir('/Users/siaga/Desktop/NFT')
files=os.listdir('/Users/siaga/Desktop/NFT')
for excel in files:
    raw=pd.read_excel(excel)
    excel=excel.replace('.xlsx','')
    excel=excel.replace('-','_')
    for column in raw:
        if column != 'm/z' and column != 'intensity':
            del raw[column]
    raw['normalized']=raw['intensity']/raw['intensity'].sum()
    del raw['intensity']
    raw=raw.rename(columns={'m/z':'mass%s'%excel,'normalized':excel})
    data=pd.concat([data,raw],axis=1,sort=False)
data.to_excel('/Users/siaga/Desktop/processedData.xlsx')