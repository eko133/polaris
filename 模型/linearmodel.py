# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:11:29 2017

@author: samuel
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression  

data = pd.read_excel("G:\\Seafile\\临时\\1.xlsx")
feature_cols = ['Toc']
x = data[feature_cols]
y = data['intensity']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
linreg = LinearRegression()
linreg.fit(x_train, y_train)
print (linreg.intercept_)
print (linreg.coef_)