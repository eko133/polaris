# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017

@author: samuel
"""
#加载所需模块，pandas提供excel支持，matplotlib.pyplot提供plt支持
import pandas as pd
import os
import numpy as np
#从excel中加载excel文件,目录自行修改
n=0
fignum=0
variation_column=pd.DataFrame(np.arange(16*18).reshape(16,18)).astype(float)
while n<18:
    os.chdir("G:\Seafile\临时\Biodegradation of sulfur-rich oil\负离子excel")
    if os.path.isfile(str(n)+'.xlsx') == True:
        data = pd.read_excel(str(n)+'.xlsx')
        data = data[data['class']=='O4S1']
        data['intensity']=data['intensity'].astype(float)
        data = data[(data.ppm>-2) & (data.ppm<2)]
        DBE_count=0
        while DBE_count<15:
            x=data[data['DBE']==DBE_count]
            variation_column.loc[DBE_count,n]=x['intensity'].sum()
            DBE_count=DBE_count+1
        sum=variation_column[n].sum()
        variation_column[n]=variation_column[n]/sum
        n=n+1
    else:
        n=n+1