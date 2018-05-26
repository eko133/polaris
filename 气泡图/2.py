# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def readAllExcel(path):
    excelFilePath=[]
    for root,dirs,files in os.walk(path):
        for excel in files:
            if os.path.splitext(excel)[1] == '.xlsx':
                excelFilePath.append(path+'/'+excel)
    return excelFilePath


os.chdir(r"D:\MQ\EXCEL")
excelFile=readAllExcel(r"D:\MQ\EXCEL")
species=['O1','O2','O3','O4','O5','N1','N1O1','N1O2','N1O3']
fignum=0
for specie in species:
    if os.path.exists(specie)==False:
        os.makedirs(specie)
for excel in excelFile:
    data=pd.read_excel(excel)
    data=data[data['DBE']>0]
    excelName=os.path.split(excel)[1]
    excelName=os.path.splitext(excelName)[0]
    
    data['intensity']=data['intensity'].astype(float)
    for specie in species:
        data_specie=data[data['class']==specie]
        sum=data_specie['intensity'].sum()
        data_specie['normalized']=data_specie['intensity']/sum
        plt.figure(fignum)
        plt.figure(figsize=(6,5))
        font = {'family' : 'arial',  
                'color'  : 'black',  
                'weight' : 'normal',  
                'size'   : 20,  
                } 
        plt.axis([0,50,0,25])
        plt.xlabel("Carbon Number",fontdict=font)
        plt.ylabel("DBE",fontdict=font)
        plt.xticks(fontsize=16,fontname='arial')
        plt.yticks(np.arange(0,26,5),fontsize=16,fontname='arial')
        plt.text(1,23,s=specie,fontdict=font)
        plt.text(43,23,s=excelName,fontdict=font)
        plt.scatter(data_specie['C'],data_specie['DBE'],s=1000*data_specie['normalized'],edgecolors='black',linewidth=0.1)
        path=r"D:\MQ\EXCEL"+"\\"+specie
        filename=excelName+'.png'
        plt.savefig(os.path.join(path,filename),dpi=600)
        fignum=fignum+1

