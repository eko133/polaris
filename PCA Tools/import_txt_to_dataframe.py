import pandas as pd
import numpy as np


with open(r'/Users/siaga/test.txt','r+') as file:
    data = file.read()
    data = data.split(';')
    data = pd.DataFrame(np.array(data).reshape((-1,3)),columns=['m/z','I','S/N'])
    data = data.astype(float)
    data.to_clipboard()



