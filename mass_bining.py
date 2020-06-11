import pandas as pd
import numpy as np
import concurrent.futures

def mass_bining(packed_args):
    lines,datan = packed_args
    for line in lines:
        data = line.split(';')
        sample_name = data[0]
        print(sample_name)
        del data[0]
        del data[0]
        tmp = pd.DataFrame(np.array(data).reshape((-1, 3)), columns=['m/z', sample_name, 'S/N'])
        tmp = tmp.drop(columns='S/N')
        tmp = tmp.astype(float)
        tmp['m/z'] = tmp['m/z'].round(2)
        tmp =tmp.groupby('m/z').max()
        datan = datan.merge(tmp, how='outer', left_index=True, right_index=True)
    return datan


mass = [i for i in np.arange(30000,60000,1)/100]

test_txt = r'/Users/siaga/Documents/gdgt/sbb_sterol.txt'
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
    data_dict[0], data_dict[1],data_dict[2],data_dict[3],data_dict[4], data_dict[5],data_dict[6],data_dict[7] =executor.map(mass_bining, args)
for i in range(0,n_proc):
    data = data.merge(data_dict[i], how='outer', left_index=True, right_index=True)



