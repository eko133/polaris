import pandas as pd
import numpy as np
import concurrent.futures
from main import mp_mass_bining


mass = [i for i in np.arange(37500,52500,1)/100]

test_txt = r'/Users/siaga/Seafile/Documents/MALDI/SBB0-5_sterol.txt'
with open(test_txt) as f:
    lines = f.readlines()
n_proc = 8
line_dict = dict()
data_dict = dict()
data = pd.DataFrame()
step = int(len(lines)/n_proc)+1
for i in range(0,n_proc):
    line_dict[i] = lines[i*step:(i+1)*step]
for i in range(0,n_proc):
    data_dict[i] = pd.DataFrame(index=mass)
args = ((line_dict[i],data_dict[i]) for i in range(0,n_proc))
with concurrent.futures.ProcessPoolExecutor() as executor:
    data_dict[0], data_dict[1],data_dict[2],data_dict[3],data_dict[4], data_dict[5],data_dict[6],data_dict[7] =executor.map(mp_mass_bining, args)
for i in range(0,n_proc):
    data = data.merge(data_dict[i], how='outer', left_index=True, right_index=True)



