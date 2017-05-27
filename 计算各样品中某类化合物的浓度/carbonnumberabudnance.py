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
std_list=[316137184,325182048,303093312,238507440,188629136,225205872,216088384,218022192,244597184,1,232608720,209771232,210359648,223166784,218055088,1,226342880,216615344]
std_1=[888471808,1,676252608,704567360,683203776,5,580047808,7,639557376,9,10,11,12,630770368,14,15,16,631176448]
variation_column = pd.DataFrame().astype(float)
variation_column.index=variation_column.index+10
while n<18:
    os.chdir("G:\Seafile\临时\Biodegradation of sulfur-rich oil\正离子excel")
    if os.path.isfile(str(n)+'.xlsx') == True:
        data = pd.read_excel(str(n)+'.xlsx')
        data = data[(data.ppm>-2) & (data.ppm<2)]
        data['intensity']=data['intensity'].astype(float)
        sum = data['intensity'].sum()
        data = data[data['class']=='S1']
        data = data[data.DBE == 1]
        CN_count=10
        while CN_count<45:
            x=data[data.C == CN_count]
            variation_column.loc[CN_count,n]=x['intensity'].sum()
            CN_count=CN_count+1
        variation_column[n]=variation_column[n]/sum
        n=n+1
    else:
        n=n+1