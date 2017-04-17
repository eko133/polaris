# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
#×Ô¶¯²éÕÒÎÄ¼þ-Ö»»æÖÆ×Ô¼ºÐèÒªµÄ»¯ºÏÎïµÄÆøÅÝÍ¼
#¼ÓÔØËùÐèÄ£¿é£¬pandasÌá¹©excelÖ§³Ö£¬matplotlib.pyplotÌá¹©pltÖ§³Ö
import pandas as pd
import matplotlib.pyplot as plt
import os
n=0
fignum=0
while n<18:
    os.chdir("G:\Seafile\临时\Biodegradation of sulfur-rich oil\负离子excel")
    if os.path.exists(str(n))==False:
        os.makedirs(str(n))
    if os.path.isfile(str(n)+'.xlsx') == True:
        data = pd.read_excel(str(n)+'.xlsx')
        data['intensity']=data['intensity'].astype(float)
        data = data[(data.ppm>-2) & (data.ppm<2)]
        specie='O2'
        x=data[data['class']==specie]
        x['normalized']=x['intensity']/x['intensity'].sum()
        plt.figure(fignum)
        font = {'family' : 'serif',  
                'color'  : 'black',  
                'weight' : 'normal',  
                'size'   : 14,  
                } 
        plt.axis([0,60,0,16])
        plt.xlabel("Carbon Number",fontdict=font)
        plt.ylabel("DBE",fontdict=font)
        plt.text(1,14,s=specie,fontdict=font)
        plt.text(53,14,s=str(n)+'w',fontdict=font)
        plt.scatter(x['C'],x['DBE'],s=3000*x['normalized'],edgecolors='white',alpha=0.8)
        path="G:\Seafile\临时\Biodegradation of sulfur-rich oil\负离子excel"+"\\"+str(n)
        filename=specie+'.png'
        plt.savefig(os.path.join(path,filename),dpi=600)
        fignum=fignum+1
        n=n+1
    else:
        n=n+1