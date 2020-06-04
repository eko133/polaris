# -*- coding: utf-8 -*-

import sys
sys.path.append(r'../')
import targeting
import concurrent.futures
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from itertools import combinations
import concurrent.futures
import math


# targeting.find_grouped_samples()

# os.chdir(r'/Users/siaga/Git/polaris/data/grouped_by_ccat_sterol')
# files = os.listdir(r'/Users/siaga/Git/polaris/data/grouped_by_ccat_sterol')
# for file in files:
#     if file.endswith('.txt'):
#         name = os.path.splitext(file)[0]
#         targeting.align(file)
#         targeting.normalizer(name + r'.csv')
#         targeting.delete_exclude_mass('normalized_'+name + '.csv')
#         targeting.multi_linear_regression('normalized_'+name + '.csv')



def linear_regression(data):
    for x, y in combinations(data.columns,2):
        ## not carbon isotopes and not carbon clusters
        difference = float(x) - float(y)
        difference_dem = math.modf(difference)[0]
        if (abs(difference) >= 2) & (abs(difference_dem)>0.01):
            print(x,y)
            tmp = data[[x,y]].copy()
            tmp = tmp.dropna(how='any', axis=0)
            if len(tmp) >= 50:

                X_train, X_test, y_train, y_test = train_test_split(tmp[x], tmp[y], test_size=0.2, random_state=0)
                LR = linear_model.LinearRegression()
                LR.fit(X_train, y_train)
                if LR.score(X_test, y_test) >=0.5:
                    with open ('./shared.txt','a') as f:
                        f.writelines(f'{x,y}, {LR.coef_}, {LR.intercept_}, {LR.score(X_test, y_test)} \n')


data = pd.read_csv('/Users/siaga/Git/polaris/data/grouped_by_ccat_sterol/normalized_ccat0.50394.csv')
data.drop(data.columns[0], axis=1, inplace=True)
data = data.set_index(data.columns[0])
# Slicing dataframe according to column values
data_sliced = {}
length = len(data.columns)
n = 8
step = int(length/n) +1
for column_start in range(0,length,step):
    print(column_start)
    data_sliced[column_start] = data.iloc[:,column_start:(column_start+step)]

# Internal regression
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     executor.map(linear_regression,list(data_sliced.values()))
for x, y in combinations(data_sliced[0].columns,2):
    ## not carbon isotopes and not carbon clusters
    difference = float(x) - float(y)
    difference_dem = math.modf(difference)[0]
    if (abs(difference) >= 2) & (abs(difference_dem)>0.01):
        print(x,y)
        tmp = data_sliced[0][[x,y]].copy()
        tmp = tmp.dropna(how='any', axis=0)
        if len(tmp) >= 50:

            X_train, X_test, y_train, y_test = train_test_split(tmp[x], tmp[y], test_size=0.2, random_state=0)
            LR = linear_model.LinearRegression()
            LR.fit(X_train, y_train)
            if LR.score(X_test, y_test) >=0.5:
                with open ('./shared.txt','a') as f:
                    f.writelines(f'{x,y}, {LR.coef_}, {LR.intercept_}, {LR.score(X_test, y_test)} \n')