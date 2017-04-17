# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
#自动查找文件-只绘制自己需要的化合物的气泡图
#加载所需模块，pandas提供excel支持，matplotlib.pyplot提供plt支持
import pandas as pd
import matplotlib.pyplot as plt
import os
#从excel中加载excel文件,目录自行修改
n=1
fignum=0
while n<18:
    os.chdir("G:\Seafile\临时\Biodegradation of sulfur-rich oil\负离子excel")
    if os.path.exists(str(n))==False:
        os.makedirs(str(n))
    data = pd.read_excel(str(n)+'.xlsx')
    #将intensity转换为float类型
    data['intensity']=data['intensity'].astype(float)
    #按ppm筛选所需数据
    data = data[(data.ppm>-2) & (data.ppm<2)]
    #遍历上述操作找到的所有化合物类，分别绘制图谱
    specie='O1'
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
    plt.scatter(x['C'],x['DBE'],s=3000*x['normalized'],edgecolors='white',alpha=0.8)
    #保存图片
    path="G:\Seafile\临时\Biodegradation of sulfur-rich oil\负离子excel"+"\\"+str(n)
    filename=specie+'.png'
    plt.savefig(os.path.join(path,filename),dpi=600)
    fignum=fignum+1
    n=n+1