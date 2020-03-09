# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt
import sys
import json
import os
sys.path.append(r'../')
import targeting

targeting.group_by_ccat()
targeting.find_grouped_samples()

os.chdir(r'/Users/siaga/Git/polaris/data/grouped_by_ccat')
files = os.listdir(r'/Users/siaga/Git/polaris/data/grouped_by_ccat')
for file in files:
    if file.endswith('.txt'):
        name = os.path.splitext(file)[0]
        targeting.align(file)
        targeting.normalizer(name + r'.csv')
        targeting.delete_exclude_mass('normalized_'+name + '.csv')
        targeting.linear_regression('normalized_'+name + '.csv')