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
        
class ParametersInput:
    def __init__(self,master,entry_name,var_name,row,column):
        self.entry_name=entry_name
        self.var_name=var_name
        self.row=row
        self.column=column
        
        self.label=Label(master, text=self.entry_name).grid(row=self.row, column=self.column)
        self.entry=Entry(master,textvariable=self.var_name)
        self.entry.grid(row=self.row, column=self.column+1)
        
    def getVar(self):
        return self.entry.get()
        
        
class RawDataButton(MenuBar):
    
    def __init__(self,parent,SignalToNoiseRatio, ppm, N, O, S,row,column):
        self.row=row
        self.column=column
        self.SignalToNoiseRatio=SignalToNoiseRatio
        self.ppm = ppm
        self.N=N
        self.O=O
        self.S=S
        
        self.button=Button(parent, text='Process raw data', command = self.ProcessRawData).grid(row=self.row,column=self.column)
    
    def ProcessRawData(self):
        global data
        print(self.SignalToNoiseRatio)
        #data = data[data['S/N']>self.SignalToNoiseRatio]
        
                
class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        menubar = MenuBar(self)
        self.config(menu=menubar)

        SignalToNoiseRatio_var=StringVar()
        ppm_var=StringVar()
        N_var=StringVar()
        S_var=StringVar()
        O_var=StringVar()
        SignalToNoiseRatio = ParametersInput(self,'S/N',SignalToNoiseRatio_var,0,0)
        SignalToNoiseRatio_var=SignalToNoiseRatio.getVar()
        ppm = ParametersInput(self,'error(ppm)',ppm_var,0,2)
        N = ParametersInput(self,'N',N_var,0,4)
        S = ParametersInput(self,'S',S_var,0,6)
        O = ParametersInput(self,'O',O_var,0,8)
        
        button=RawDataButton(self,SignalToNoiseRatio_var,ppm_var,N_var,S_var,O_var,0,10)


        
if __name__ == '__main__':
    app=App()
    app.mainloop()