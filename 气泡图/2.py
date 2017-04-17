# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
#�Զ������ļ�-ֻ�����Լ���Ҫ�Ļ����������ͼ
#��������ģ�飬pandas�ṩexcel֧�֣�matplotlib.pyplot�ṩplt֧��
import pandas as pd
import matplotlib.pyplot as plt
import os
#��excel�м���excel�ļ�,Ŀ¼�����޸�
n=1
fignum=0
while n<18:
    os.chdir("G:\Seafile\��ʱ\Biodegradation of sulfur-rich oil\������excel")
    if os.path.exists(str(n))==False:
        os.makedirs(str(n))
    data = pd.read_excel(str(n)+'.xlsx')
    #��intensityת��Ϊfloat����
    data['intensity']=data['intensity'].astype(float)
    #��ppmɸѡ��������
    data = data[(data.ppm>-2) & (data.ppm<2)]
    #�������������ҵ������л������࣬�ֱ����ͼ��
    specie='O1'
    x=data[data['class']==specie]
    x['normalized']=x['intensity']/x['intensity'].sum()
    #�ֱ��ͼ
    plt.figure(fignum)
    #����ͼƬ��ʽ
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
    #����ͼƬ
    path="G:\Seafile\��ʱ\Biodegradation of sulfur-rich oil\������excel"+"\\"+str(n)
    filename=specie+'.png'
    plt.savefig(os.path.join(path,filename),dpi=600)
    fignum=fignum+1
    n=n+1