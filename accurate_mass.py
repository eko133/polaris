import pandas as pd
import numpy as np
import concurrent.futures

def accurate_mass_finder(packed_args):
    lines,target_mass, datan = packed_args
    mass = dict()
    for line in lines:
        data = line.split(';')
        sample_name = data[0]
        print(sample_name)
        del data[0]
        del data[0]
        tmp = pd.DataFrame(np.array(data).reshape((-1, 3)), columns=['m/z', sample_name, 'S/N'])
        tmp = tmp.drop(columns='S/N')
        tmp = tmp.astype(float)
        for i in range(len(target_mass)):
            try:
                tmp1 = tmp[(tmp['m/z']>=target_mass[i]-0.005) &(tmp['m/z']<=target_mass[i]+0.005)]
                mass[i] = tmp1['m/z'].max()
            except IndexError:
                mass[i] = 0
        print([mass[i] for i in range(len(target_mass))])
        datan.append([mass[i] for i in range(len(target_mass))])
    return datan


target_mass = [392.15,517.27,393.3,433.26,405.19,460.25]
test_txt = r'/Users/siaga/Dropbox/Documents/MALDI/sbb_sterol.txt'
with open(test_txt) as f:
    lines = f.readlines()
n_proc = 8
line_dict = dict()
data_dict = dict()
data=list()
step = int(len(lines)/n_proc)+1
for i in range(0,n_proc):
    line_dict[i] = lines[i*step:(i+1)*step]
for i in range(0,n_proc):
    data_dict[i] = list()
args = ((line_dict[i],target_mass, data_dict[i]) for i in range(0,n_proc))
with concurrent.futures.ProcessPoolExecutor() as executor:
    data_dict[0], data_dict[1],data_dict[2],data_dict[3],data_dict[4], data_dict[5],data_dict[6],data_dict[7] =executor.map(accurate_mass_finder, args)
for i in range(0,n_proc):
    data = data+data_dict[i]
data = pd.DataFrame(data)
target_mass= [data[i].mean() for i in range(len(target_mass))]



