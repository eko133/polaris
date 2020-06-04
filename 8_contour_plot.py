import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

figdir = r'./figure/气泡图/N1Cl1/'
if not os.path.exists(figdir):
    os.makedirs(figdir)

with open (r'./pkl/negative_ESI_result_2 (2020_04_02 06_19_35 UTC).pkl','rb') as f:
    data=pickle.load(f)
for i in data:
    data[i] = data[i][data[i]['Class'] == 'N1Cl1']
    tmp = data[i][['C','dbe','I']]
    tmp['dbe'] = tmp['dbe'].astype(int)
    tmp['dbe'] = tmp['dbe'] +1
    tmp['C'] = tmp['C'].astype(int)
    tmp = tmp[(tmp['dbe'] >=1) & (tmp['dbe'] <=25)]
    tmp = tmp[(tmp['C'] >10) & (tmp['C'] <=56)]
    try:
        tmp.loc[(tmp.dbe == 13) & (tmp.C == 33), 'I'] = tmp.loc[(tmp.dbe == 13) & (tmp.C == 32),'I'].tolist()[0]
    except:
        tmp.loc[(tmp.dbe == 13) & (tmp.C == 33), 'I'] = 0
    try:
        tmp.loc[(tmp.dbe == 14) & (tmp.C == 35), 'I'] = tmp.loc[(tmp.dbe == 14) & (tmp.C == 34),'I'].tolist()[0]
    except:
        tmp.loc[(tmp.dbe == 14) & (tmp.C == 35), 'I'] = 0
    tmp['normalized'] = (tmp['I']-tmp['I'].min())/(tmp['I'].max()-tmp['I'].min())
    del tmp['I']
    x=tmp['C'].values
    x=np.array(x,dtype=float)
    y=tmp['dbe'].values
    y=np.array(y,dtype=float)
    z=tmp['normalized'].values
    z=np.array(z,dtype=float)
    plt.figure(figsize=(6,5),dpi=300)
    plt.scatter(x,y,s=100*z)
    plt.xticks(range(10,56,5), fontsize = 16)
    plt.yticks(range(0,26,5), fontsize = 16)
    # plt.title(i, fontsize = 20)
    figfile = figdir+r'%s'%i
    plt.savefig(figfile)


