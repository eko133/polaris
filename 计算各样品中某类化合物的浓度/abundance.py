# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017

@author: samuel
"""
#加载所需模块，pandas提供excel支持，matplotlib.pyplot提供plt支持
import pandas as pd
import os
#从excel中加载excel文件,目录自行修改
n=0
fignum=0
variation_column=pd.DataFrame().astype(float)
while n<18:
    os.chdir("G:\Seafile\临时\Biodegradation of sulfur-rich oil\正离子excel")
    if os.path.isfile(str(n)+'.xlsx') == True:
        data = pd.read_excel(str(n)+'.xlsx')
        data['intensity']=data['intensity'].astype(float)
        data = data[(data.ppm>-2) & (data.ppm<2)]
        data = data[data['class'] != 'O2S2']
        sum = data['intensity'].sum()
        y=data['class']
        y=y.drop_duplicates()
        y=y.reset_index()
        specie_num=len(y['class'])
        specie_count=0
        while specie_count<specie_num:
            specie=y.loc[specie_count,'class']
            x=data[data['class']==specie]
            variation_column.loc[specie,n]=x['intensity'].sum()
            specie_count=specie_count+1
        variation_column[n]=variation_column[n]/sum
        n=n+1
    else:
        n=n+1