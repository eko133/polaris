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


compound = run.generate_possible_formula()
data = run.read_raw_csv()
for i in data:
    data[i] = run.speculate_formula(data[i],compound)
with open('./negative_ESI_result.pkl', 'wb') as f:
    pickle.dump(data,f)








