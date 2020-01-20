import pandas as pd
import os

os.chdir('/Users/siaga/Desktop/NFT')
files=os.listdir('/Users/siaga/Desktop/NFT')
excelIndex=list()
for i in files:
    excelIndex.append(i.replace('.xlsx',''))
data=pd.DataFrame(index=excelIndex)

for excel in files:
    tmp=pd.read_excel(excel)
    excel=excel.replace('.xlsx','')
    # To calculate A/C ratios (acylic to 2-4 ring O2 species ratios)
    tmp_O2=tmp[tmp['class']=='O2']
    tmp_acyclic=tmp_O2[tmp_O2['DBE']==1]
    tmp_cyclic=tmp_O2[tmp_O2['DBE']>2]
    tmp_cyclic=tmp_cyclic[tmp_cyclic['DBE']<6]
    data.loc[excel,'A/C Ratios']=tmp_acyclic['intensity'].sum()/tmp_cyclic['intensity'].sum()

    # To calculate MA ratios (O1 Dbe=4/5 and dbe=4/7)
    tmp_O1=tmp[tmp['class']=='O1']
    tmp_O1_4=tmp_O1[tmp_O1['DBE']==4]
    tmp_O1_5=tmp_O1[tmp_O1['DBE']==5]
    tmp_O1_7=tmp_O1[tmp_O1['DBE']==7]
    data.loc[excel, 'MA1'] =tmp_O1_4['intensity'].sum()/tmp_O1_5['intensity'].sum()
    data.loc[excel, 'MA2'] =tmp_O1_4['intensity'].sum()/tmp_O1_7['intensity'].sum()

    # To calculate SA index (low DBE  acids dbe=1-6)
    tmp_O2_sa=tmp_O2[tmp_O2['DBE']>0]
    tmp_O2_sa=tmp_O2_sa[tmp_O2_sa['DBE']<7]
    data.loc[excel, 'SA Index']=tmp_O2['intensity'].sum()/tmp['intensity'].sum()

data.to_excel('/Users/siaga/Desktop/ratios.xlsx')