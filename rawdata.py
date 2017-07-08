# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:52:52 2017

@author: samuel
"""
import pandas as pd
import os
import itertools

class compound:
    def __init__(self,c,h,o,n,s):
        self.c=c
        self.h=h
        self.o=o
        self.n=n
        self.s=s
        self.mw=12 * C + 14.003074 * N + 31.972071 * S + 15.9949146 * O + 1.007825 * h + 0.0005485799
        self.dbe=(2*c+1+n-h)/2
        self.km=self.mw/14.01565*14
        self.kmd=int(self.km)+1-self.km
        self.memw=0
        self.intensity=0
        
def isMolecule(a):
    if 0.3<=a.h/a.c<=3.0:
        if a.o/a.c<=3.0 and a.n/a.c<=0.5:
            if a.h<=1+2*a.c+a.n:
                if (a.h+a.n)%2 == 1:
                    if a.mw>=mw_min:
                        return True

compound_list = []

raw_data = pd.read_clipboard().astype(float)
raw_data = raw_data[raw_data['S/N'] >= 6.0]
for column in raw_data:
    if column != 'm/z' and column != 'I':
        del raw_data[column]
mw_max = raw_data['m/z'].max()
mw_min = raw_data['m/z'].min()
cmax = int(mw_max/12)
for N,S,O in itertools.product(range(6),range(6),range(6)):
    cmax = int((mw_max-14*N-16*O-32*S)/12)
    for C in range(1,cmax+1):
        hmax = int(mw_max)-12*C-14*N-16*O-32*S+1
        for H in range(hmax+1):
            molecule = compound(C,H,O,N,S)
            if isMolecule(molecule):
                boundary = 0.0015*14.01565/14
                data_test = raw_data[(raw_data['m/z']>=(molecule.mw-boundary)) & (raw_data['m/z']<=(molecule.mw+boundary))]
                if not data_test.empty:
                    molecule.intensity = data_test['I'].max()
                    data_test = data_test[data_test['I']==molecule.intensity]
                    data_test = data_test['m/z'].tolist()
                    molecule.memw = data_test[0]
                    compound_list.append(molecule)
print(len(compound_list))