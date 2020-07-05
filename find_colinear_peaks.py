import pandas as pd
from itertools import combinations
from util.preprocessing import peak_normalize
from util.preprocessing import mp_colinear_finder

from multiprocessing import Pool


if __name__ == "__main__":
    bin_data = pd.read_pickle(r'~/Downloads/alkenone_bin.pkl')
    bin_data = bin_data.T
    bin_data = bin_data.dropna(axis=1, thresh=0.01 * bin_data.shape[0])
    bin_data = peak_normalize(bin_data, normalize='none')
    mass_lists_combinations = list(combinations(bin_data.columns,2))
    args = ((bin_data,mass) for mass in mass_lists_combinations)
    with Pool(8) as p:
        p.map(mp_colinear_finder, args)

