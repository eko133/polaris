import pandas as pd
import numpy as np

data = pd.read_excel('./pixel.xlsx')
start = data.pixel_y.min()
end = data.pixel_y.max()
average_points = np.linspace(start,end,227,endpoint=True)
averaged_data = pd.DataFrame()
for i in range(len(average_points)-2):
    data_tmp = data[(data['pixel_y'] >= average_points[i]) & (data['pixel_y'] <= average_points[i+1])]
    if len(data_tmp)>= 10:
        ccat = data_tmp.ccat.mean()
        ccat_new = data_tmp.ccat_new.mean()
        y = data_tmp.pixel_y.mean()
        averaged_data.loc[y,'ccat'] = ccat
        averaged_data.loc[y,'ccat_new'] = ccat_new
averaged_data.to_excel('./averaged_data.xlsx')



