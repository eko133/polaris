import pickle
import pandas as pd

f = open('./gdgt_similarMassMerged2.pkl','rb')
samples = pickle.load(f)
samples = samples.set_index('m/z')
biomarker = pd.DataFrame(index=samples.columns)
for column in samples:
    gdgt_5 = samples.loc[1314.23,column]
    gdgt_0 = samples.loc[1324.31,column]
    biomarker.loc[column,'ccat1'] = gdgt_5/(gdgt_0+gdgt_5)
biomarker = biomarker.dropna()
biomarker.to_pickle('./ccat2.pkl')
