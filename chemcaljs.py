from execjs import get
import os
import pandas as pd
import numpy as np

with open(r'/Users/siaga/test.txt','r+') as file:
    data = file.read()
    data = data.split(';')
    data = pd.DataFrame(np.array(data).reshape((-1,3)),columns=['m/z','I','S/N'])
    data = data.astype(float)

runtime = get('Node')
context = runtime.compile('''
    module.paths.push('%s');
    chemcalc = require('chemcalc');
    function mfFromMonoisotopicMass(mass,cfg){
          return chemcalc.mfFromMonoisotopicMass(mass,cfg);  
    }
''' % os.path.join(os.path.dirname(__file__),'node_modules'))

for i in range(len(data)):
    if i < len(data):
        mass_Na = data.loc[i, 'm/z']
        result = context.call('mfFromMonoisotopicMass',mass_Na,{'mfRange':'C1-200H1-200O0-10Na+','maxUnsaturation':'10','useUnsaturation':'true','integerUnsaturation':'false','massRange':'0.005'})
        try:
            for m in range(len(result['results'])):
                if isinstance(result['results'][m]['unsat'], int):
                    continue
                else:
                    mass_iso_em = result['results'][m]['em'] - 12 + 13.003355
                    mass_iso = data[(data['m/z'] >= (mass_iso_em - 0.005)) & (data['m/z'] <= (mass_iso_em + 0.005))]
                    if not mass_iso.empty:
                        data.loc[i, 'real mass'] = result['results'][m]['em']
                        data.loc[i, 'error(ppm)'] = result['results'][m]['ppm']
                        data.loc[i, 'mf'] = result['results'][m]['mf']
                        data.loc[i, 'unsat'] = result['results'][m]['unsat']
                        data = data[data['m/z'] != mass_iso['m/z'].max()].reset_index(drop=True)
                        break
                    else:
                        continue
        except IndexError:
            data.loc[i,'real mass'] = 'nan'
    else:
        break
data= data.dropna(axis=0)


