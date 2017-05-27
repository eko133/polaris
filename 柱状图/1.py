# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:28:00 2017

@author: samuel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("C:\\Users\\samuel\\Desktop\\1.xlsx")
fig, ax = plt.subplots()
count_row = data.shape[0]
count_col = data.shape[1]
bar_width = 0.35
opacity = 0.8
i = 0
While i<count_col+1:
    
    