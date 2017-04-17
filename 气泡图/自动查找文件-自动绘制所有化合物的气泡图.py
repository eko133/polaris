# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
#加载所需模块，pandas提供excel支持，matplotlib.pyplot提供plt支持
import pandas as pd
import matplotlib.pyplot as plt
import os
#从excel中加载excel文件,目录自行修改
n=1
fignum=0
while n<18:
    os.chdir("G:\Seafile\临时\Biodegradation of sulfur-rich oil\负离子excel")
    os.makedirs(str(n))
    data = pd.read_excel(str(n)+'.xlsx')
    #将intensity转换为float类型
    data['intensity']=data['intensity'].astype(float)
    #按ppm筛选所需数据
    data = data[(data.ppm>-2) & (data.ppm<2)]
    #读取数据的所有化合物类，先剔除掉重复项，再将剩下的列举出来
    y=data['class']
    y=y.drop_duplicates()
    y=y.reset_index()
    m=len(y)
    i=0
    specie=0
    #遍历上述操作找到的所有化合物类，分别绘制图谱
    while i<m:
        specie=y.loc[i,'class']
        x=data[data['class']==specie]
        x['normalized']=x['intensity']/x['intensity'].sum()
        #分别绘图
        plt.figure(fignum)
        #设置图片格式
        font = {'family' : 'serif',  
                'color'  : 'black',  
                'weight' : 'normal',  
                'size'   : 14,  
                } 
        plt.axis([0,60,0,16])
        plt.xlabel("Carbon Number",fontdict=font)
        plt.ylabel("DBE",fontdict=font)
        plt.text(1,14,s=specie,fontdict=font)
        plt.scatter(x['C'],x['DBE'],s=1200*x['normalized'])
        sample_file_name = specie
        #保存图片
        path="G:\Seafile\临时\Biodegradation of sulfur-rich oil\负离子excel"+"\\"+str(n)
        filename=specie+'.png'
        plt.savefig(os.path.join(path,filename),dpi=600)
        i=i+1
        fignum=fignum+1
    n=n+1