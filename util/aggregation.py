import pandas as pd
import numpy as np

def average(data,target,spacing):
    """
    0.08 spacing for monthly resolution
    0.25 spacing for seasonally resolution
    exlude biomarker ratio equals to 0 or 1
    """
    data = data[['age',target]]
    data[target] = data[target].replace(1,np.nan)
    data[target] = data[target].replace(0,np.nan)
    data = data.dropna()
    start = data.age.min()
    end = data.age.max()
    average_points = np.arange(start, end, spacing)
    for i in range(len(average_points)-1):
        if len(data[data.age.between(average_points[i],average_points[i+1])]) >= 10:
            if i == len(average_points)-2:
                data.loc[data.age>=average_points[i+1],'age'] = average_points[i+1]
            data.loc[data.age.between(average_points[i],average_points[i+1]),'age'] = average_points[i]
        else:
            data.loc[data.age.between(average_points[i], average_points[i + 1]), 'age'] = np.nan
    data = data.dropna()
    data = data.groupby('age').mean()
    return data