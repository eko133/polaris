# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017

@author: samuel
"""
#加载所需模块，pandas提供excel支持，matplotlib.pyplot提供plt支持
import pandas as pd
import os
#从excel中加载excel文件,目录自行修改
n=1
variation_column=pd.DataFrame().astype(float)
data = pd.read_excel(r'G:\Gdrive\Archived\Biodegradation of sulfur-rich oil\负离子excel\6.xlsx')
data['intensity']=data['intensity'].astype(float)
sum = data['intensity'].sum()
data = data[data['class']=='O1']
data['intensity']=data['intensity'].astype(float)
DBE_count=0
while DBE_count<20:
    x=data[data['DBE']==DBE_count]
    variation_column.loc[DBE_count,n]=x['intensity'].sum()
    DBE_count=DBE_count+1
variation_column[n]=variation_column[n]/sum
