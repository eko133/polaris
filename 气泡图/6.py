# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
n=0
fignum=0
while n<18:
    os.chdir("G:\Seafile\临时\Biodegradation of sulfur-rich oil\正离子excel")
    if os.path.exists(str(n))==False:
        os.makedirs(str(n))
    if os.path.isfile(str(n)+'.xlsx') == True:
        data = pd.read_excel(str(n)+'.xlsx')
        data['intensity']=data['intensity'].astype(float)
        data = data[(data.ppm>-2) & (data.ppm<2)]
        sum = data['intensity'].sum()
        data = data[(data['class'] == 'S1') | (data['class'] == 'S2')| (data['class'] == 'S3')| (data['class'] == 'S4')]
        data['H'] = 2*(data['C']+1-data['DBE'])
        data['H/C'] = data['H']/data['C']
        data['S'] = data['class'].str.get(1)
        data['S'] = data['S'].astype(float)
        data['S/C'] = data['S']/data['C']
        data['normalized'] = data['intensity']/sum
        plt.figure(fignum)
        font = {'family' : 'serif',  
                'color'  : 'black',  
                'weight' : 'normal',  
                'size'   : 14,  
                } 
        plt.axis([0,0.2,0,2.1])
        plt.xlabel("S/C",fontdict=font)
        plt.ylabel("H/C",fontdict=font)
        plt.text(0.165,1.9,s='Z-'+str(n),fontdict=font)
        norm = matplotlib.colors.Normalize(vmax=0.0005)
        plt.scatter(data['S/C'],data['H/C'],s=5,c=data['normalized'],cmap='jet',norm=norm,edgecolors='none')
        plt.colorbar()
        path="G:\Seafile\临时\Biodegradation of sulfur-rich oil\正离子excel"+"\\"+str(n)
        filename=str(n)+'S.png'
        plt.savefig(os.path.join(path,filename), dpi=600)
        fignum=fignum+1
        n=n+1
    else:
        n=n+1