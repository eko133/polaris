# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:25:42 2017

@author: samuel
"""

from tkinter import *
from tkinter import ttk
import pandas as pd
from tkinter import filedialog
import os
import itertools

atomic_mass = {'C':12.0107, 'H':1.007825, 'N':14.003074, 'O':15.9949146, 'S':31.972071, 'e':0.0005485799, 'CH2':14.01565}
mass_tolerance=0.0015*14.01565/14

class Compound:
    def __init__(self,c,h,n,o,s):
        self.c=c
        self.h=h
        self.o=o
        self.n=n
        self.s=s
        self.mw=12 * C + atomic_mass['N'] * N + atomic_mass['S'] * S + atomic_mass['O'] * O + atomic_mass['H']* h + atomic_mass['e']
        self.dbe=(2*c+1+n-h)/2
        self.km=self.mw/atomic_mass['CH2']*14
        self.kmd=int(self.km)+1-self.km
        self.measured_mw=0
        self.intensity=0 

def isMolecule(a):
    if 0.3<=a.h/a.c<=3.0:
        if a.o/a.c<=3.0 and a.n/a.c<=0.5:
            if a.h<=1+2*a.c+a.n:
                if (a.h+a.n)%2 == 1:
                    if a.mw>=mw_min:
                        return True

class MenuBar(Menu):
        
    def __init__(self,parent):
        Menu.__init__(self,parent)
        
        fileMenu=Menu(self)
        self.add_cascade(label='File', menu=fileMenu)
        fileMenu.add_command(label='import from clipboard', command=self.readClipboard)
        fileMenu.add_command(label='import from excel', command=self.readExcel)
        fileMenu.add_command(label='import from folder',command=self.readFolder)
        
    def readClipboard(self):
        global data
        data = pd.read_clipboard().astype(float)
    
    def readExcel(self):
        global data
        excel_path=filedialog.askopenfilename(defaultextension='.xlsx', filetypes=(('Excel', '*.xlsx'), ('2003 Excel', '*.xls'), ('CSV', '*.csv'), ('All Files', '*.*')))
        if os.path.splitext(excel_path)[1] == '.xlsx' or 'xls':
            data = pd.read_excel(excel_path).astype(float)
        elif os.path.splitext(excel_path)[1] == '.csv':
            data = pd.read_csv(excel_path).astype(float)
    
    def readFolder(self):
        global folder_path
        folder_path=filedialog.askdirectory()
        
class topFrame:
    
    def __init__(self,parent):
        self.frame=Frame(parent)
        self.frame.pack()
        
        self.snLabel=Label(self.frame, text='S/N')
        self.snLabel.grid(row=0,column=0)
        self.snEntry=Entry(self.frame)
        self.snEntry.grid(row=0,column=1)
        
        self.ppmLabel=Label(self.frame, text='error(ppm)')
        self.ppmLabel.grid(row=0,column=2)
        self.ppmEntry=Entry(self.frame)
        self.ppmEntry.grid(row=0,column=3)
        
        self.nLabel=Label(self.frame, text='N')
        self.nLabel.grid(row=0,column=4)
        self.nEntry=Entry(self.frame)
        self.nEntry.grid(row=0,column=5)
        
        self.oLabel=Label(self.frame, text='O')
        self.oLabel.grid(row=0,column=6)
        self.oEntry=Entry(self.frame)
        self.oEntry.grid(row=0,column=7)
        
        self.sLabel=Label(self.frame, text='S')
        self.sLabel.grid(row=0,column=8)
        self.sEntry=Entry(self.frame)
        self.sEntry.grid(row=0,column=9)
        
        self.processButton=Button(self.frame, text='OK and process data', command=self.processData)
        self.processButton.grid(row=0,column=10)
        
    def processData(self):
        compound_list = []
        global data
        data = data[data['S/N']>=int(self.snEntry.get())]
        for column in data:
            if column != 'm/z' and column != 'I':
                del data[column]
        mw_max=data['m/z'].max()
        mw_min=data['m/z'].min()
        for N,O,S in itertools.product(range(int(self.nEntry.get())+1), range(int(self.oEntry.get())+1),range(int(self.sEntry.get())+1)):
            c_max=int((mw_max-14*N-16*O-32*S)/12)
            for C in range(1,c_max+1):
                h_max=int(mw_max)-12*C-14*N-16*O-32*S+1
                for H in range(1,h_max+1):
                    molecule=Compound(C,H,N,O,S)
                    if isMolecule(molecule):
                        data_test=data[[(data['m/z']>=(molecule.mw-mass_tolerance)) & (data['m/z']<=(molecule.mw+mass_tolerance))]]
                        if not data_test.empty:
                            molecule.intensity = data_test['I'].max()
                            data_test = data_test[data_test['I']==molecule.intensity]
                            data_test = data_test['m/z'].tolist()
                            molecule.memw = data_test[0]
                            compound_list.append(molecule)
                
class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        menubar = MenuBar(self)
        self.config(menu=menubar)
        frame =topFrame(self)

        
if __name__ == '__main__':
    app=App()
    app.mainloop()