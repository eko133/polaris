import pandas as pd
import numpy as np
import concurrent.futures
from mendeleev import element
import itertools
import json
import main




compound = main.generate_possible_formula()

test_txt = r'/Users/siaga/Seafile/Documents/MALDI/SBB0-5_alkenone.txt'
with open(test_txt) as f:
    lines = f.readlines()
data = lines[1].split(';')
sample_name = data[0]
print(sample_name)
del data[0]
del data[0]
tmp = pd.DataFrame(np.array(data).reshape((-1, 3)), columns=['m/z', 'I', 'S/N'])
tmp = tmp.drop(columns='S/N')
tmp = tmp.astype(float)
for key in compound:
    key_min, key_max = key-0.005, key+0.005
    tmp1 = tmp[(tmp['m/z']>=key_min) & (tmp['m/z']<=key_max)]
    try:
        id = tmp1.idxmax()[1]
        tmp.loc[id,'em'] = key
    except ValueError:
        continue
# tmp=tmp.dropna()
tmp['ppm'] = 1000000*(tmp['m/z']-tmp['em'])/tmp['m/z']

# n_proc = 8
# line_dict = dict()
# data_dict = dict()
# data = pd.DataFrame()
# step = int(len(lines)/n_proc)+1
# for i in range(0,n_proc):
#     line_dict[i] = lines[i*step:(i+1)*step]
# for i in range(0,n_proc):
#     data_dict[i] = pd.DataFrame()
# args = ((line_dict[i],data_dict[i]) for i in range(0,n_proc))
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     data_dict[0], data_dict[1],data_dict[2],data_dict[3],data_dict[4], data_dict[5],data_dict[6],data_dict[7] =executor.map(mass_bining, args)
# for i in range(0,n_proc):
#     data = data.merge(data_dict[i], how='outer', left_index=True, right_index=True)
