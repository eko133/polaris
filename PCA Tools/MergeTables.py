import pandas as pd
import os
import sys

#data=pd.read_excel(sys.argv[1])
data=pd.read_excel('/Users/siaga/Desktop/processedData.xlsx')
mass=set()
basket=pd.DataFrame()
excelList=os.listdir('/Users/siaga/Desktop/NFT')
for i in excelList:
    i=i.replace('.xlsx','')
    i=i.replace('-','_')
    locals()['data'+i]=data[['mass'+i,i]].dropna()
    mass.update(data['mass'+i])
mass = {x for x in mass if pd.notna(x)}
for m in excelList:
    m=m.replace('.xlsx','')
    m=m.replace('-','_')
    locals()['masse'+m]=mass-set(data['mass'+m])
    locals()['masse'+m]=pd.DataFrame(locals()['masse'+m],columns=['mass'+m])
    locals()['data'+m]=pd.concat([locals()['data'+m],locals()['masse'+m]],ignore_index=True,sort=False).fillna(0)
    locals()['data'+m]=locals()['data'+m].sort_values(by=['mass'+m])
    locals()['data' + m]=locals()['data'+m].reset_index(drop=True)
    basket=pd.concat([basket,locals()['data'+m]],axis=1,sort=False)
basket=basket.rename(columns={'massL5_450':"mtoz"})
for column in basket:
    if 'mass' in column:
        del basket[column]
basket[['mtoz','formula']]=basket['mtoz'].str.split(',',expand=True)
basket['mtoz']=basket['mtoz'].astype(float)
basket=basket.sort_values(by=['mtoz'])
basket.to_excel('/Users/siaga/Desktop/pca_processed.xlsx',index=False)

# data.to_excel('/Users/siaga/Desktop/pca_processed.xlsx')

# masse1=pd.DataFrame(masse1,columns=['mass1'])
# data1=pd.concat([data1,masse1],ignore_index=True,sort=False).fillna(0)
# data1=data1.sort_values(by=['mass1'])



# masse2=pd.DataFrame(masse2,columns=['mass2'])
# masse3=pd.DataFrame(masse3,columns=['mass3'])
# masse4=pd.DataFrame(masse4,columns=['mass4'])
# masse5=pd.DataFrame(masse5,columns=['mass5'])
# masse6=pd.DataFrame(masse6,columns=['mass6'])
# data=pd.concat([data,masse1,masse2,masse3,masse4,masse5,masse6],ignore_index=True,sort=False)
# data.to_excel('/Users/siaga/Desktop/pca_processed.xlsx')