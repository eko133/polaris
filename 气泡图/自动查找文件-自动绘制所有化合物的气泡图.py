# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
#��������ģ�飬pandas�ṩexcel֧�֣�matplotlib.pyplot�ṩplt֧��
import pandas as pd
import matplotlib.pyplot as plt
import os
#��excel�м���excel�ļ�,Ŀ¼�����޸�
n=1
fignum=0
while n<18:
    os.chdir("G:\Seafile\��ʱ\Biodegradation of sulfur-rich oil\������excel")
    os.makedirs(str(n))
    data = pd.read_excel(str(n)+'.xlsx')
    #��intensityת��Ϊfloat����
    data['intensity']=data['intensity'].astype(float)
    #��ppmɸѡ��������
    data = data[(data.ppm>-2) & (data.ppm<2)]
    #��ȡ���ݵ����л������࣬���޳����ظ���ٽ�ʣ�µ��оٳ���
    y=data['class']
    y=y.drop_duplicates()
    y=y.reset_index()
    m=len(y)
    i=0
    specie=0
    #�������������ҵ������л������࣬�ֱ����ͼ��
    while i<m:
        specie=y.loc[i,'class']
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
        plt.scatter(x['C'],x['DBE'],s=1200*x['normalized'])
        sample_file_name = specie
        #����ͼƬ
        path="G:\Seafile\��ʱ\Biodegradation of sulfur-rich oil\������excel"+"\\"+str(n)
        filename=specie+'.png'
        plt.savefig(os.path.join(path,filename),dpi=600)
        i=i+1
        fignum=fignum+1
    n=n+1