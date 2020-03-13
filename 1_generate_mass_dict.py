import pandas as pd
from mendeleev import element
import itertools
import run
import numpy as np
import sys
import os
from multiprocessing import cpu_count, Pool
from functools import partial
import ast
import pickle

def parallelize(self, data, func):
    compound = self.compound
    compound_split = np.array_split(compound, cores)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, compound_split))
    pool.close()
    pool.join()
    return data

if __name__ == '__main__':
    compound = run.generate_possible_formula()
    data = run.read_raw_csv()
    cores = cpu_count()
    data1 =data['L0_330']







