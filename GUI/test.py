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
from tkinter import messagebox


atomic_mass = {'C':12.0107, 'H':1.007825, 'N':14.003074, 'O':15.9949146, 'S':31.972071, 'e':0.0005485799, 'CH2':14.01565}
mass_tolerance=0.0015*14.01565/14

class Compound:
    def __init__(self,c,h,n,o,s,mode):
        self.c=c
        self.h=h
        self.o=o
        self.n=n
        self.s=s
        self.mode=mode
        self.mw=self.mw()
        self.dbe=(2*c+1+n-h)/2
        self.km=self.mw/atomic_mass['CH2']*14
        self.kmd=int(self.km)+1-self.km
        self.memw=0
        self.intensity=0
        self.ppm = 0
        self.specie=self.specie()
        
    def mw(self):
        if self.mode=='+':
            a=12 * self.c + atomic_mass['N'] *self. n + atomic_mass['S'] *self. s + atomic_mass['O'] * self.o + atomic_mass['H']*self. h - atomic_mass['e']
        elif self.mode=='-':
            a=12 * self.c + atomic_mass['N'] *self. n + atomic_mass['S'] *self. s + atomic_mass['O'] * self.o + atomic_mass['H']*self. h + atomic_mass['e']
        return a
    
    def specie(self):
        specie_n=''
        specie_o=''
        specie_s=''
        if self.n!=0:
            specie_n='N'+'%d'%self.n 
        if self.o!=0:
            specie_o='O'+'%d'%self.o 
        if self.s!=0:
            specie_s='S'+'%d'%self.s 
        specie_all=specie_n+specie_o+specie_s
        return specie_all
            

def isMolecule(a,mw_min):
    if a.n!=0 or a.o!=0 or a.s!=0:
        if 0.3<=a.h/a.c<=3.0:
            if a.o/a.c<=3.0 and a.n/a.c<=0.5:
                if a.h<=1+2*a.c+a.n:
                    if (a.h+a.n)%2 == 1:
                        if a.mw>=mw_min:
                            return True

def excelSave(excelFile):
    path=filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=(('Excel', '*.xlsx'), ('2003 Excel', '*.xls'),('All Files', '*.*')))
    if path is None:
        return
    else:
        writer=pd.ExcelWriter(path, engine = 'xlsxwriter')
    excelFile.to_excel(writer,'Sheet1')
    writer.save()
        

class MenuBar(Menu):
        
    def __init__(self,parent):
        Menu.__init__(self,parent)
        
        fileMenu=Menu(self)
        self.add_cascade(label='File', menu=fileMenu)
        fileMenu.add_command(label='import from clipboard', command=self.readClipboard)
        fileMenu.add_command(label='import from excel', command=self.readExcel)
        fileMenu.add_command(label='import from folder',command=self.readFolder)
        
        self.text_widget=parent.text_widget
        
        self.data=pd.DataFrame()
        self.folderpath=StringVar()
        
    def readClipboard(self):
        self.data = pd.read_clipboard().astype(float)
        self.text_widget.delete('1.0',END)
        self.text_widget.insert(END,self.data)
        
    def readExcel(self):
        excel_path=filedialog.askopenfilename(defaultextension='.xlsx', filetypes=(('Excel', '*.xlsx'), ('2003 Excel', '*.xls'), ('CSV', '*.csv'), ('All Files', '*.*')))
        if os.path.splitext(excel_path)[1] == '.xlsx' or 'xls':
            self.data = pd.read_excel(excel_path).astype(float)
        elif os.path.splitext(excel_path)[1] == '.csv':
            self.data = pd.read_csv(excel_path).astype(float)
    
    def readFolder(self):
        self.folder_path=filedialog.askdirectory()
                
class topFrame:
    
    def __init__(self,parent,menubar):
        self.frame=Frame(parent)
        self.frame.pack()
        
        self.menubar=menubar
                
        self.snLabel=Label(self.frame, text='S/N')
        self.snLabel.grid(row=0,column=0)
        self.snEntry=Entry(self.frame)
        self.snEntry.insert(END,'6')
        self.snEntry.grid(row=0,column=1)
        
        self.ppmLabel=Label(self.frame, text='error(ppm)')
        self.ppmLabel.grid(row=0,column=2)
        self.ppmEntry=Entry(self.frame)
        self.ppmEntry.insert(END,'1.2')
        self.ppmEntry.grid(row=0,column=3)
        
        self.nLabel=Label(self.frame, text='N')
        self.nLabel.grid(row=0,column=4)
        self.nEntry=Entry(self.frame)
        self.nEntry.insert(END,'5')
        self.nEntry.grid(row=0,column=5)
        
        self.oLabel=Label(self.frame, text='O')
        self.oLabel.grid(row=0,column=6)
        self.oEntry=Entry(self.frame)
        self.oEntry.insert(END,'5')
        self.oEntry.grid(row=0,column=7)
        
        self.sLabel=Label(self.frame, text='S')
        self.sLabel.grid(row=0,column=8)
        self.sEntry=Entry(self.frame)
        self.sEntry.insert(END,'5')
        self.sEntry.grid(row=0,column=9)
        
        self.modeLabel=Label(self.frame, text='ESI mode(+,-)')
        self.modeLabel.grid(row=0,column=10)
        self.modeEntry=Entry(self.frame)
        self.modeEntry.insert(END,'+')
        self.modeEntry.grid(row=0,column=11)
        
        self.processButton=Button(self.frame, text='OK and process data', command=self.processData)
        self.processButton.grid(row=0,column=12)
        
        self.text_widget=parent.text_widget
        
        
    def processData(self):
        
        saveExcel=pd.DataFrame()
        for i in ('m/z','ppm','class','C','H','O','N','S','intensity'):
            saveExcel.loc[0,i]=i
            i+=i
        count=0
        self.data=self.menubar.data
        self.data = self.data[self.data['S/N']>=int(self.snEntry.get())]
        for column in self.data:
            if column != 'm/z' and column != 'I':
                del self.data[column]
        mw_max=self.data['m/z'].max()
        mw_min=self.data['m/z'].min()
        for N,O,S in itertools.product(range(int(self.nEntry.get())+1), range(int(self.oEntry.get())+1),range(int(self.sEntry.get())+1)):
            c_max=int((mw_max-14*N-16*O-32*S)/12)
            for C in range(1,c_max+1):
                h_max=int(mw_max)-12*C-14*N-16*O-32*S+1
                for H in range(1,h_max+1):
                    molecule=Compound(C,H,N,O,S,self.modeEntry.get())
                    if isMolecule(molecule,mw_min):
                        data_test=self.data[(self.data['m/z']>=(molecule.mw-mass_tolerance)) & (self.data['m/z']<=(molecule.mw+mass_tolerance))]
                        if not data_test.empty:
                            molecule.intensity = data_test['I'].max()
                            data_test = data_test[data_test['I']==molecule.intensity]
                            data_test = data_test['m/z'].tolist()
                            molecule.memw = data_test[0]
                            molecule.ppm = abs(1000000*(molecule.mw-molecule.memw)/molecule.mw)
                            if molecule.ppm <= float(self.ppmEntry.get()):
                                stringTovar={'m/z':molecule.mw,'ppm':molecule.ppm,'class':molecule.specie,'C':molecule.c,'H':molecule.h,'O':molecule.o,'N':molecule.n,'S':molecule.s,'intensity':molecule.intensity}
                                for column in saveExcel:
                                    saveExcel.loc[count,column]=stringTovar[column]
                                count+=1
        self.text_widget.delete('1.0',END)
        self.text_widget.insert(END,saveExcel)
        excelSave(saveExcel)
        
        
class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.text_widget = Text(self)
        self.text_widget.pack()
        menubar = MenuBar(self)
        self.config(menu=menubar)
        
        
        frame =topFrame(self,menubar)

        
if __name__ == '__main__':
    app=App()
    app.mainloop()