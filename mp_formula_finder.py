import pandas as pd
import concurrent.futures
import main

# for alkenone dataset
# mass_calibration_list = [553.5319,551.5162,567.5475,559.4274,573.4067,569.4540,551.4435,557.2523,555.2367]
# for sterol dataset
mass_calibration_list = [433.4380, 503.4799, 505.4955, 453.3703, 441.2975, 433.3805, 493.4016, 409.3441, 411.3597, 421.3441, 435.3597, 437.3754, 439.3910, 451.3910,475.4486, 477.4642 ]

compound = main.generate_possible_formula(375,525)

test_txt = r'/Users/siaga/Seafile/Documents/MALDI/SBB0-5_sterol.txt'
with open(test_txt) as f:
    lines = f.readlines()
lines = lines[:20]
n_proc = 8
line_dict = dict()
data_dict = dict()
data = pd.DataFrame()
step = int(len(lines)/n_proc)+1
for i in range(0,n_proc):
    line_dict[i] = lines[i*step:(i+1)*step]
for i in range(0,n_proc):
    data_dict[i] = pd.DataFrame()
args = ((line_dict[i],data_dict[i],mass_calibration_list, compound) for i in range(0,n_proc))
with concurrent.futures.ProcessPoolExecutor() as executor:
    data_dict[0], data_dict[1],data_dict[2],data_dict[3],data_dict[4], data_dict[5],data_dict[6],data_dict[7] =executor.map(main.mp_align_with_recalibration, args)
for i in range(0,n_proc):
    data = data.merge(data_dict[i], how='outer', left_index=True, right_index=True)

