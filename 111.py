import pandas as pd
import os
import numpy as np

tmp = os.listdir(r'/Users/siaga/Git/polaris/data/grouped_by_ccat')
file = list()
samples = {}
basket = pd.DataFrame()
for i in tmp:
    if 'lr'  in i:
        file.append(i)
for csv in file:
    ccat = os.path.splitext(csv)[0]
    ccat = ccat.replace('lr_normalized_ccat','')
    samples[ccat] =pd.read_csv(r'/Users/siaga/Git/polaris/data/grouped_by_ccat/'+csv)
    del samples[ccat]['Intercept']
    del samples[ccat]['Score']
    samples[ccat].set_index(samples[ccat].columns[0],inplace=True)
    samples[ccat].rename(columns={'Coef':ccat},inplace=True)
for ccat in samples:
    basket = basket.merge(samples[ccat], how ='outer', left_index=True, right_index=True)
basket = basket.reset_index()
basket = basket.replace('nan',np.nan)
basket = basket.dropna(thresh=10)
basket.to_csv(r'./merged_regression.csv')
