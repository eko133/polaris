# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
n=0
fignum=0
while n<18:
    os.chdir(r"G:\Gdrive\Archived\Biodegradation of sulfur-rich oil\负离子excel")
    if os.path.exists(str(n))==False:
        os.makedirs(str(n))
    if os.path.isfile(str(n)+'.xlsx') == True:
        data = pd.read_excel(str(n)+'.xlsx')
        data['intensity']=data['intensity'].astype(float)
        data = data[(data.ppm>-2) & (data.ppm<2)]
        data = data[(data['DBE']>0) ]
        specie='O2S3'
        x=data[data['class']==specie]
        sum=x['intensity'].sum()
        x['normalized']=x['intensity']/sum
        
        plt.figure(fignum)
        plt.figure(figsize=(6,5))
        font = {'family' : 'arial',  
                'color'  : 'black',  
                'weight' : 'normal',  
                'size'   : 20,  
                } 
        plt.axis([0,60,0,16])
        plt.xlabel("Carbon Number",fontdict=font)
        plt.ylabel("DBE",fontdict=font)
        plt.xticks(fontsize=16,fontname='arial')
        plt.yticks(fontsize=16,fontname='arial')
        plt.text(1,14,s=specie,fontdict=font)
        plt.text(53,14,s='Z-'+str(n),fontdict=font)
        plt.scatter(x['C'],x['DBE'],s=500*x['normalized'],edgecolors='black',linewidth=0.1)
        path=r"G:\Gdrive\Archived\Biodegradation of sulfur-rich oil\负离子excel"+"\\"+str(n)
        filename=specie+'.png'
        plt.savefig(os.path.join(path,filename),dpi=1000)
        fignum=fignum+1
        n=n+1
    else:
        n=n+1