import pandas as pd
import sys

#data=pd.read_excel(sys.argv[1])
data=pd.read_excel('/Users/siaga/Desktop/pca.xlsx')
data1=data.iloc[:,0:2].dropna()
data2=data.iloc[:,2:4].dropna()
data3=data.iloc[:,4:6].dropna()
data4=data.iloc[:,6:8].dropna()
data5=data.iloc[:,8:10].dropna()
data6=data.iloc[:,10:12].dropna()
mass=set()
for i in range(1,7):
    mass.update(data['mass%d'%i])
mass = {x for x in mass if pd.notna(x)}
#for i in range(1,12,2):
#    locals()['df'+str(int((i+1)/2))]=data.iloc[:,i-1:i+1]
for m in range(1,7):
    locals()['masse'+str(m)]=mass-set(data['mass%d'%m])
    locals()['masse'+str(m)]=pd.DataFrame(locals()['masse'+str(m)],columns=['mass%d'%m])
    locals()['data'+str(m)]=pd.concat([locals()['data'+str(m)],locals()['masse'+str(m)]],ignore_index=True,sort=False).fillna(0)
    locals()['data'+str(m)]=locals()['data'+str(m)].sort_values(by=['mass%d'%m])
    locals()['data' + str(m)]=locals()['data'+str(m)].reset_index(drop=True)
data=pd.concat([data1,data2,data3,data4,data5,data6],axis=1,sort=False)
data.to_excel('/Users/siaga/Desktop/pca_processed.xlsx')

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