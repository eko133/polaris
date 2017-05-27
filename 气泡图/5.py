# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:32:50 2017

@author: samuel
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:06:23 2017
@author: samuel
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import griddata
n=0
fignum=0
std_list=[316137184,325182048,303093312,238507440,188629136,225205872,216088384,218022192,244597184,1,232608720,209771232,210359648,223166784,218055088,1,226342880,216615344]
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
        x['normalized']=x['intensity']/std_list[n]
        plt.figure(fignum)
        font = {'family' : 'serif',  
                'color'  : 'black',  
                'weight' : 'normal',  
                'size'   : 14,  
                } 
        X,Y = np.meshgrid(x['C'],x['DBE'])
        Z = griddata((x['C'],x['DBE']),x['normalized'],(X,Y),method='nearest')
        plt.contourf(X,Y,Z,cmap=plt.cm.hot)
        
        #plt.contourf(X,Y,x['normalized'])
        
        
        
        
        plt.axis([0,60,0,16])
        plt.xlabel("Carbon Number",fontdict=font)
        plt.ylabel("DBE",fontdict=font)
        plt.text(1,14,s=specie,fontdict=font)
        plt.text(53,14,s=str(n)+'w',fontdict=font)
        path="G:\Seafile\临时\Biodegradation of sulfur-rich oil\负离子excel"+"\\"+str(n)
        filename=specie+'.png'
        plt.savefig(os.path.join(path,filename),dpi=600)
        fignum=fignum+1
        n=n+1
    else:
        n=n+1