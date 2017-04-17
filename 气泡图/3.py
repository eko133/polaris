# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
#×Ô¶¯²éÕÒÎÄ¼þ-×Ô¶¯»æÖÆËùÓÐ»¯ºÏÎïµÄÆøÅÝÍ¼
#¼ÓÔØËùÐèÄ£¿é£¬pandasÌá¹©excelÖ§³Ö£¬matplotlib.pyplotÌá¹©pltÖ§³Ö
import pandas as pd
import matplotlib.pyplot as plt
import os
#´ÓexcelÖÐ¼ÓÔØexcelÎÄ¼þ,Ä¿Â¼×ÔÐÐÐÞ¸Ä
n=1
fignum=0
while n<18:
    os.chdir("G:\Seafile\ÁÙÊ±\Biodegradation of sulfur-rich oil\负离子excel")
    os.makedirs(str(n))
    data = pd.read_excel(str(n)+'.xlsx')
    #½«intensity×ª»»ÎªfloatÀàÐÍ
    data['intensity']=data['intensity'].astype(float)
    #°´ppmÉ¸Ñ¡ËùÐèÊý¾Ý
    data = data[(data.ppm>-2) & (data.ppm<2)]
    #¶ÁÈ¡Êý¾ÝµÄËùÓÐ»¯ºÏÎïÀà£¬ÏÈÌÞ³ýµôÖØ¸´Ïî£¬ÔÙ½«Ê£ÏÂµÄÁÐ¾Ù³öÀ´
    y=data['class']
    y=y.drop_duplicates()
    y=y.reset_index()
    m=len(y)
    i=0
    specie=0
    #±éÀúÉÏÊö²Ù×÷ÕÒµ½µÄËùÓÐ»¯ºÏÎïÀà£¬·Ö±ð»æÖÆÍ¼Æ×
    while i<m:
        specie=y.loc[i,'class']
        x=data[data['class']==specie]
        x['normalized']=x['intensity']/x['intensity'].sum()
        #·Ö±ð»æÍ¼
        plt.figure(fignum)
        #ÉèÖÃÍ¼Æ¬¸ñÊ½
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
        #±£´æÍ¼Æ¬
        path="G:\Seafile\ÁÙÊ±\Biodegradation of sulfur-rich oil\¸ºÀë×Óexcel"+"\\"+str(n)
        filename=specie+'.png'
        plt.savefig(os.path.join(path,filename),dpi=600)
        i=i+1
        fignum=fignum+1
    n=n+1