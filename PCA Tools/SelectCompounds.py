import pandas as pd
import os

os.chdir('/Users/siaga/Desktop/NFT')
files=os.listdir('/Users/siaga/Desktop/NFT')
excelIndex=list()
for i in files:
    excelIndex.append(i.replace('.xlsx',''))
data=pd.DataFrame(index=excelIndex)
N1_9=list(range(14,20))
N1_12=list(range(16,22))
N1_15=list(range(20,26))
O2_1=list(range(19,28))
for excel in files:
    tmp=pd.read_excel(excel)
    excel=excel.replace('.xlsx','')
    tmp_N1=tmp[tmp['class']=='N1']
    tmp_N1_12=tmp_N1[tmp_N1['DBE']==12]
    tmp_N1_15=tmp_N1[tmp_N1['DBE']==15]
    tmp_O2=tmp[tmp['class']=='O2']
    tmp_O2_1=tmp_O2[tmp_O2['DBE']==1]
    for i in N1_12:
        try:
            data.loc[excel,"%sN1_12"%i]=tmp_N1_12.loc[tmp_N1_12['C']==i,'intensity'].item()
        except ValueError:
            data.loc[excel, "%sN1_12"%i]=0
    for i in N1_15:
        try:
            data.loc[excel,"%sN1_15"%i]=tmp_N1_15.loc[tmp_N1_15['C']==i,'intensity'].item()
        except ValueError:
            data.loc[excel,"%sN1_15"%i]=0
    for i in O2_1:
        try:
            data.loc[excel,"%sO2_1"%i]=tmp_O2_1.loc[tmp_O2_1['C']==i,'intensity'].item()
        except ValueError:
            data.loc[excel,"%sO2_1"%i]=0
data.to_excel('/Users/siaga/Desktop/importantcompounds.xlsx')