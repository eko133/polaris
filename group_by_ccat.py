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
sys.path.append(r'../')
import targeting


targeting.LR('./ccat0.47393.csv')