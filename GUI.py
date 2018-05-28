# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:25:42 2017

@author: samuel
"""

from tkinter import *
from tkinter import ttk
import pandas as pd
from tkinter import filedialog
from tkinter import simpledialog
import os
import itertools
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np

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
        self.realh=self.realh()
        self.dbe=(2*c+2+n-self.realh)/2
        self.km=self.mw/atomic_mass['CH2']*14
        self.kmd=int(self.km)+1-self.km
        self.memw=0
        self.intensity=0
        self.ppm = 0
        self.specie=self.specie()
        
    def mw(self):
        if self.mode==1 or 3:
            a=12 * self.c + atomic_mass['N'] *self. n + atomic_mass['S'] *self. s + atomic_mass['O'] * self.o + atomic_mass['H']*self. h - atomic_mass['e']
        elif self.mode==2:
            a=12 * self.c + atomic_mass['N'] *self. n + atomic_mass['S'] *self. s + atomic_mass['O'] * self.o + atomic_mass['H']*self. h + atomic_mass['e']
        return a

    def realh(self):
        if self.mode==1 or 3:
            b=self.h-1
        if self.mode==2:
            b=self.h+1
        return b
        
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
        if specie_all=='':
            specie_all='CH'
        if self.dbe != int(self.dbe):
            specie_all='*'+specie_all
        return specie_all
            

def isMolecule(a,mw_min):
    if a.mode==1 or 2:
        if a.n!=0 or a.o!=0 or a.s!=0:
            if 0.3<=a.h/a.c<=3.0:
                if a.o/a.c<=3.0 and a.n/a.c<=0.5:
                    if a.h<=1+2*a.c+a.n:
                        if (a.h+a.n)%2 == 1:
                            if a.mw>=mw_min:
                                return True
    if a.mode==3:
        if 0.3<=a.h/a.c<=3.0:
            if a.o/a.c<=3.0 and a.n/a.c<=0.5:
                if a.h<=3+2*a.c+a.n:
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

def readAllExcel(path):
    excelFilePath=[]
    for root,dirs,files in os.walk(path):
        for excel in files:
            if os.path.splitext(excel)[1] == '.xlsx':
                excelFilePath.append(path+'/'+excel)
    return excelFilePath

class ParaDialog(Toplevel):
    
    def __init__(self):
        Toplevel.__init__(self)
        self.title('PARAMETERS')
        self.para=[]
        self.setup_UI()
        
    def setup_UI(self):      
        row1=Frame(self)
        row1.pack()
        Label(row1,text='C',width=5).pack(side=LEFT)
        self.cacstart=IntVar()
        self.cacstop=IntVar()
        Entry(row1,textvariable=self.cacstart,width=5).pack(side=LEFT)
        Label(row1,text='–').pack(side=LEFT)
        Entry(row1,textvariable=self.cacstop,width=5).pack(side=LEFT)
        
        Label(row1,text='DBE',width=5).pack(side=LEFT)
        self.cadbe=IntVar()
        Entry(row1, textvariable=self.cadbe,width=5).pack(side=LEFT)
                
        Label(row1,text='Mode',width=5).pack(side=LEFT)
        self.camo=IntVar()
        Radiobutton(row1,text='+ESI',variable=self.camo,value=1).pack(side=LEFT)
        Radiobutton(row1,text='-ESI',variable=self.camo,value=2).pack(side=LEFT)
        
        row2=Frame(self)
        row2.pack()
        Label(row2,text='N',width=5).pack(side=LEFT)
        self.can=IntVar()
        Entry(row2, textvariable=self.can,width=5).pack(side=LEFT)
        
        Label(row2,text='O',width=5).pack(side=LEFT)
        self.cao=IntVar()
        Entry(row2, textvariable=self.cao,width=5).pack(side=LEFT)
        
        Label(row2,text='S',width=5).pack(side=LEFT)
        self.cas=IntVar()
        Entry(row2, textvariable=self.cas,width=5).pack(side=LEFT)
        
        row3=Frame(self)
        row3.pack()
        Button(row3,text='Cancel',command=self.cancel).pack(side=RIGHT)
        
        Label(row3, text='\t').pack(side=RIGHT)
        
        Button(row3,text='OK',command=self.ok).pack(side=RIGHT)
    
    def ok(self):
        self.para=[self.cacstart.get(),self.cacstop.get(),self.cadbe.get(),self.camo.get(),self.can.get(),self.cao.get(),self.cas.get()]
        self.destroy()
        
    def cancel(self):
        self.para=None
        self.destroy()

class MenuBar(Menu):       
    
    def __init__(self,parent,rawdataframe,bubbleplotframe):
        Menu.__init__(self,parent)
        
        fileMenu=Menu(self)
        
        self.bubbleplotframe=bubbleplotframe
        self.rawdataframe=rawdataframe
        
        self.add_cascade(label='Import', menu=fileMenu)
        fileMenu.add_command(label='From clipboard', command=self.readClipboard)
        fileMenu.add_command(label='From file', command=self.readExcel)
        fileMenu.add_command(label='From folder',command=self.readFolder)
        
        self.text_widget=parent.text_widget
        
        self.data=pd.DataFrame().astype(float)
        self.folder_path=0
        self.excelName=0
        
        calMenu=Menu(self)
        self.add_cascade(label='Calculate', menu=calMenu)
        calMenu.add_command(label='Possible formulas', command=self.processData)
        calMenu.add_command(label='Molecular weight calibration', command=self.mwCa)
        calMenu.add_command(label='Class abundance from file', command=self.calAbundance)
        calMenu.add_command(label='Class abundance from folder', command=self.calAbundanceFile)
        calMenu.add_command(label='Class DBE abundance from file', command=self.caldbeAbundance)
        calMenu.add_command(label='Class DBE abundance from folder', command=self.caldbeAbundanceFile)
        
        plotMenu=Menu(self)
        self.add_cascade(label='Plot', menu=plotMenu)
        plotMenu.add_command(label='Bar plot from file', command=self.barplot)
        plotMenu.add_command(label='Bubble plots from file', command=self.bubbleplotfile)
        plotMenu.add_command(label='Bubble plots from folder', command=self.bubbleplot)
        
        aboutMenu=Menu(self)
        self.add_cascade(label='Help', menu=aboutMenu)
        aboutMenu.add_command(label='About', command=self.aboutMessage)
        
        
        
        
    def readClipboard(self):
        self.data = pd.read_clipboard()
        if not ('m/z' or 'I' or 'S/N') in self.data.columns:
            messagebox.showerror('Error',"Wrong data format!\nMUST include 'm/z', 'I', and 'S/N'")
        self.excelName='Clipboard'
        self.text_widget.delete('1.0',END)
        self.text_widget.insert(END,self.data)
        
    def readExcel(self):
        excel_path=filedialog.askopenfilename(defaultextension='.xlsx', filetypes=(('Excel', '*.xlsx'), ('2003 Excel', '*.xls'), ('CSV', '*.csv'), ('All Files', '*.*')))
        if os.path.splitext(excel_path)[1] == '.xlsx' or 'xls':
            self.data = pd.read_excel(excel_path)
        elif os.path.splitext(excel_path)[1] == '.csv':
            self.data = pd.read_csv(excel_path)
        excelName=os.path.splitext(excel_path)[0]
        self.excelName=excelName.split('\\')[-1]
        self.text_widget.delete('1.0',END)
        self.text_widget.insert(END,self.data)
        
    def readFolder(self):
        self.folder_path=filedialog.askdirectory()
        self.text_widget.delete('1.0',END)
        path=readAllExcel(self.folder_path)
        self.text_widget.insert(END,'These are the Excels found in the path: \n')
        for paths in path:
            self.text_widget.insert(END,paths+'\n')
         
    def calAbundance(self):
        try:
            data=self.data
            if not 'normalized' in data.columns:
                data['normalized']=data['intensity']/data['intensity'].sum()        
            species=data['class']
            species=species.drop_duplicates()
            abundance=pd.DataFrame().astype(float)
            for specie in species:
                data_specie=data[data['class'] == specie]
                abundance.loc[specie,self.excelName] = data_specie['normalized'].sum()
            self.text_widget.delete('1.0',END)
            self.text_widget.insert(END,abundance)
            excelSave(abundance)
        except:
            messagebox.showerror('Error', 'Please import data first!')

    def setPara(self):
        para=self.askPara()
        if para is None: return
        
        self.cacstart, self.cacstop,self.cadbe,self.camo,self.can,self.cao,self.cas=para
        
        
    def askPara(self):
        inputDialog=ParaDialog()
        self.wait_window(inputDialog)
        return inputDialog.para

    def processData(self):
        if self.rawdataframe.modeEntry.get()==3:
            self.processAPPIData()
        self.processESIData()
        
    def processESIData(self):
        
        saveExcel=pd.DataFrame()
        for i in ('measured m/z','m/z','ppm','class','C','H','O','N','S','DBE','intensity'):
            saveExcel.loc[0,i]=i
            i+=i
        count=0
        self.data = self.data[self.data['S/N']>=int(self.rawdataframe.snEntry.get())]
        for column in self.data:
            if column != 'm/z' and column != 'I':
                del self.data[column]
        mw_max=self.data['m/z'].max()
        mw_min=self.data['m/z'].min()
        for N,O,S in itertools.product(range(int(self.rawdataframe.nEntry.get())+1), range(int(self.rawdataframe.oEntry.get())+1),range(int(self.rawdataframe.sEntry.get())+1)):
            c_max=int((mw_max-14*N-16*O-32*S)/12)
            for C in range(1,c_max+1):
                h_max=int(mw_max)-12*C-14*N-16*O-32*S+1
                for H in range(1,h_max+1):
                    molecule=Compound(C,H,N,O,S,self.rawdataframe.modeEntry.get())
                    if isMolecule(molecule,mw_min):
                        data_test=self.data[(self.data['m/z']>=(molecule.mw-mass_tolerance)) & (self.data['m/z']<=(molecule.mw+mass_tolerance))]
                        if not data_test.empty:
                            molecule.intensity = data_test['I'].max()
                            data_test = data_test[data_test['I']==molecule.intensity]
                            data_test = data_test['m/z'].tolist()
                            molecule.memw = data_test[0]
                            molecule.ppm = abs(1000000*(molecule.mw-molecule.memw)/molecule.mw)
                            if molecule.ppm <= float(self.rawdataframe.ppmEntry.get()):
                                stringTovar={'measured m/z':molecule.memw,'m/z':molecule.mw,'ppm':molecule.ppm,'class':molecule.specie,'C':molecule.c,'H':molecule.realh,'O':molecule.o,'N':molecule.n,'S':molecule.s,'DBE':molecule.dbe,'intensity':molecule.intensity}
                                for column in saveExcel:
                                    saveExcel.loc[count,column]=stringTovar[column]
                                count+=1
        self.text_widget.delete('1.0',END)
        self.text_widget.insert(END,saveExcel)
        excelSave(saveExcel)


    def processAPPIData(self):
        
        saveExcel=pd.DataFrame()
        for i in ('measured m/z','m/z','ppm','class','C','H','O','N','S','DBE','intensity'):
            saveExcel.loc[0,i]=i
            i+=i
        count=0
        self.data = self.data[self.data['S/N']>=int(self.rawdataframe.snEntry.get())]
        for column in self.data:
            if column != 'm/z' and column != 'I':
                del self.data[column]
        mw_max=self.data['m/z'].max()
        mw_min=self.data['m/z'].min()
        for N,O,S in itertools.product(range(int(self.rawdataframe.nEntry.get())+1), range(int(self.rawdataframe.oEntry.get())+1),range(int(self.rawdataframe.sEntry.get())+1)):
            c_max=int((mw_max-14*N-16*O-32*S)/12)
            for C in range(1,c_max+1):
                h_max=int(mw_max)-12*C-14*N-16*O-32*S+1
                for H in range(1,h_max+1):
                    molecule=Compound(C,H,N,O,S,self.rawdataframe.modeEntry.get())
                    if isMolecule(molecule,mw_min):
                        data_test=self.data[(self.data['m/z']>=(molecule.mw-mass_tolerance)) & (self.data['m/z']<=(molecule.mw+mass_tolerance))]
                        if not data_test.empty:
                            molecule.intensity = data_test['I'].max()
                            data_test = data_test[data_test['I']==molecule.intensity]
                            data_test = data_test['m/z'].tolist()
                            molecule.memw = data_test[0]
                            molecule.ppm = abs(1000000*(molecule.mw-molecule.memw)/molecule.mw)
                            if molecule.ppm <= float(self.rawdataframe.ppmEntry.get()):
                                stringTovar={'measured m/z':molecule.memw,'m/z':molecule.mw,'ppm':molecule.ppm,'class':molecule.specie,'C':molecule.c,'H':molecule.realh,'O':molecule.o,'N':molecule.n,'S':molecule.s,'DBE':molecule.dbe,'intensity':molecule.intensity}
                                for column in saveExcel:
                                    saveExcel.loc[count,column]=stringTovar[column]
                                count+=1
        self.text_widget.delete('1.0',END)
        self.text_widget.insert(END,saveExcel)
        excelSave(saveExcel)


    def mwCa(self):
        self.setPara()
        capath=filedialog.asksaveasfilename(defaultextension='ref')
        caref=open(capath,'a')
        if self.camo==1:
            for cac in range(self.cacstart,self.cacstop):
                camolecule=Compound(cac,2*cac+3+self.can-2*self.cadbe,self.can,self.cao,self.cas,1)
                caformula='C'+str(cac)+'H'+str(camolecule.h)
                if not self.can==0:
                    caformula=caformula+'N'+str(camolecule.n)
                if not self.cao==0:
                    caformula=caformula+'O'+str(camolecule.o)
                if not self.cas==0:
                    caformula=caformula+'S'+str(camolecule.s)
                caref.write(caformula+' '+str(camolecule.mw)+' '+'1+')   
                caref.write('\n')
                
        if self.camo==2:
            for cac in range(self.cacstart,self.cacstop):
                camolecule=Compound(cac,2*cac+1+self.can-2*self.cadbe,self.can,self.cao,self.cas,2)
                caformula='C'+str(cac)+'H'+str(camolecule.h)
                if not self.can==0:
                    caformula=caformula+'N'+str(camolecule.n)
                if not self.cao==0:
                    caformula=caformula+'O'+str(camolecule.o)
                if not self.cas==0:
                    caformula=caformula+'S'+str(camolecule.s)
                caref.write(caformula+' '+str(camolecule.mw)+' '+'1-')   
                caref.write('\n')
        caref.close()
        
    def calAbundanceFile(self):
        try: 
            excelFile=readAllExcel(self.folder_path)
            abundance=pd.DataFrame().astype(float)
            for excel in excelFile:
                self.excelName=os.path.split(excel)[1]
                self.excelName=os.path.splitext(self.excelName)[0]
                self.data=pd.read_excel(excel)
                data=self.data
                if not 'normalized' in data.columns:
                    data['normalized']=data['intensity']/data['intensity'].sum()        
                species=data['class']
                species=species.drop_duplicates()
                for specie in species:
                    data_specie=data[data['class'] == specie]
                    abundance.loc[specie,self.excelName] = data_specie['normalized'].sum()
                self.text_widget.delete('1.0',END)
                self.text_widget.insert(END,abundance)
            excelSave(abundance)
        except:
            messagebox.showerror('Error', 'Please import data first!')
            
    def caldbeAbundance(self):
        try: 
            data=self.data
            if not 'normalized' in data.columns:
                data['normalized']=data['intensity']/data['intensity'].sum()        
            species=data['class']
            species=species.drop_duplicates()
            abundance=pd.DataFrame().astype(float)
            dbe = 0 
            for specie in species:
                data_specie=data[data['class'] == specie]
                for dbe in range(0,20):
                    data_dbe=data_specie[data_specie['DBE'] == dbe]
                    abundance.loc[specie,dbe] = data_dbe['normalized'].sum()
            self.text_widget.delete('1.0',END)
            self.text_widget.insert(END,abundance)
            excelSave(abundance)
        except:
            messagebox.showerror('Error', 'Please import data first!')
 
    def caldbeAbundanceFile(self):
        try:
            excelFile=readAllExcel(self.folder_path)
            for excel in excelFile:
                self.excelName=excel
                self.data=pd.read_excel(excel)
                self.caldbeAbundance()
        except:
            messagebox.showerror('Error', 'Please import data first!')
    
    def barplot(self):
        try:
            data=self.data
            plt.figure(figsize=(15,10))
            plt.bar(data.index,data.iloc[:,0],align='center', alpha=0.5)
            plt.show()
        except:
            messagebox.showerror('Error', 'Please import data first!')
            
    def bubbleplotfile(self):
        species=self.bubbleplotframe.bpclass.get()
        excelName=os.path.split(self.excelName)[1]
        species=species.split(',')
        data=self.data
        data=data[data['DBE']>0]
        data['intensity']=data['intensity'].astype(float)
        path=filedialog.askdirectory()
        for specie in species:
            data_specie=data[data['class']==specie]
            sum=data_specie['intensity'].sum()
            data_specie['normalized']=data_specie['intensity']/sum
            plt.figure(figsize=(6,5))
            font = {'family' : 'arial',  
                    'color'  : 'black',  
                    'weight' : 'normal',  
                    'size'   : 20,  
                    } 
            plt.axis([int(self.bubbleplotframe.bpcstart.get()),int(self.bubbleplotframe.bpcstop.get()),int(self.bubbleplotframe.bpdbestart.get()),int(self.bubbleplotframe.bpdbestop.get())])
            plt.xlabel("Carbon Number",fontdict=font)
            plt.ylabel("DBE",fontdict=font)
            plt.xticks(fontsize=16,fontname='arial')
            plt.yticks(np.arange(int(self.bubbleplotframe.bpdbestart.get()),int(self.bubbleplotframe.bpdbestop.get())+1,2),fontsize=16,fontname='arial')
            if self.bubbleplotframe.bpshowc.get()==0:
                plt.text(int(self.bubbleplotframe.bpcstart.get())+1,int(self.bubbleplotframe.bpdbestop.get())-3,s=specie,fontdict=font)
            if self.bubbleplotframe.bpshows.get()==0:
                plt.text(int(self.bubbleplotframe.bpcstop.get())-5,int(self.bubbleplotframe.bpdbestop.get())-3,s=excelName,fontdict=font)
            plt.scatter(data_specie['C'],data_specie['DBE'],s=float(self.bubbleplotframe.bpscale.get())*data_specie['normalized'],edgecolors='black',linewidth=0.1)
            filename=specie+'.png'
            plt.savefig(os.path.join(path,filename),dpi=1000)
        messagebox.showinfo("Complete!", "All plots are stored successfully!")
        
    def bubbleplot(self):
        os.chdir(self.folder_path)
        excelFile=readAllExcel(self.folder_path)
        species=self.bubbleplotframe.bpclass.get()
        species=species.split(',')
        for specie in species:
            if os.path.exists(specie)==False:
                os.makedirs(specie)
        for excel in excelFile:
            data=pd.read_excel(excel)
            data=data[data['DBE']>0]
            excelName=os.path.split(excel)[1]
            excelName=os.path.splitext(excelName)[0]
            data['intensity']=data['intensity'].astype(float)
            for specie in species:
                data_specie=data[data['class']==specie]
                sum=data_specie['intensity'].sum()
                data_specie['normalized']=data_specie['intensity']/sum
                plt.figure(figsize=(6,5))
                font = {'family' : 'arial',  
                        'color'  : 'black',  
                        'weight' : 'normal',  
                        'size'   : 20,  
                        } 
                plt.axis([int(self.bubbleplotframe.bpcstart.get()),int(self.bubbleplotframe.bpcstop.get()),int(self.bubbleplotframe.bpdbestart.get()),int(self.bubbleplotframe.bpdbestop.get())])
                plt.xlabel("Carbon Number",fontdict=font)
                plt.ylabel("DBE",fontdict=font)
                plt.xticks(fontsize=16,fontname='arial')
                plt.yticks(np.arange(int(self.bubbleplotframe.bpdbestart.get()),int(self.bubbleplotframe.bpdbestop.get())+1,2),fontsize=16,fontname='arial')
                if self.bubbleplotframe.bpshowc.get()==0:
                    plt.text(int(self.bubbleplotframe.bpcstart.get())+1,int(self.bubbleplotframe.bpdbestop.get())-3,s=specie,fontdict=font)
                if self.bubbleplotframe.bpshows.get()==0:
                    plt.text(int(self.bubbleplotframe.bpcstop.get())-5,int(self.bubbleplotframe.bpdbestop.get())-3,s=excelName,fontdict=font)
                plt.scatter(data_specie['C'],data_specie['DBE'],s=float(self.bubbleplotframe.bpscale.get())*data_specie['normalized'],edgecolors='black',linewidth=0.1)
                path=self.folder_path+"\\"+specie
                filename=excelName+'.png'
                plt.savefig(os.path.join(path,filename),dpi=600)
        messagebox.showinfo("Complete!", "All plots are stored in the same folder with excels")

    def aboutMessage(self):
        messagebox.showinfo(title='About', message='FT–ICR MS Data Handler\nLicensed under the terms of the Apache License 2.0\n\nDeveloped and maintained by Weimin Liu\n\nFor bug reports and feature requests, please go to my Github website')
        
class RawDataFrame:
    
    def __init__(self,parent):
        self.frame=Frame(parent)
        self.frame.pack()
        
        
        Label(self.frame, text='RAW DATA',width=10).pack(side=LEFT)
                
        Label(self.frame, text='S/N',width=3).pack(side=LEFT)
        self.snEntry=Entry(self.frame,width=3)
        self.snEntry.insert(END,'6')
        self.snEntry.pack(side=LEFT)
        
        Label(self.frame, text='Error',width=5).pack(side=LEFT)
        self.ppmEntry=Entry(self.frame,width=3)
        self.ppmEntry.insert(END,'1.2')
        self.ppmEntry.pack(side=LEFT)
        
        Label(self.frame, text='N',width=3).pack(side=LEFT)
        self.nEntry=Entry(self.frame,width=3)
        self.nEntry.insert(END,'5')
        self.nEntry.pack(side=LEFT)
        
        Label(self.frame, text='O',width=3).pack(side=LEFT)
        self.oEntry=Entry(self.frame,width=3)
        self.oEntry.insert(END,'5')
        self.oEntry.pack(side=LEFT)
        
        Label(self.frame, text='S',width=3).pack(side=LEFT)
        self.sEntry=Entry(self.frame,width=3)
        self.sEntry.insert(END,'5')
        self.sEntry.pack(side=LEFT)
        

        Label(self.frame, text='Source',width=5).pack(side=LEFT)

        self.modeEntry=IntVar()
        
        Radiobutton(self.frame,text='+ESI',variable=self.modeEntry,value=1).pack(side=LEFT)
        Radiobutton(self.frame,text='-ESI',variable=self.modeEntry,value=2).pack(side=LEFT)
        Radiobutton(self.frame,text='APPI',variable=self.modeEntry,value=3).pack(side=LEFT)
#        Radiobutton(self.frame,text='-APPI',variable=self.modeEntry,value=4).pack(side=LEFT)

                        
        
class BubblePlotFrame:
    
    def __init__(self,parent):
        self.frame=Frame(parent)
        self.frame.pack()
                
        Label(self.frame, text='BUBBLE PLOT',width=12).pack(side=LEFT)
        
        Label(self.frame, text='C', width=3).pack(side=LEFT)
        
        self.bpcstart=Entry(self.frame,width=3)
        self.bpcstart.insert(END,'10')
        self.bpcstart.pack(side=LEFT)      
        
        Label(self.frame, text='–', width=3).pack(side=LEFT)
        
        self.bpcstop=Entry(self.frame,width=3)
        self.bpcstop.insert(END,'50')
        self.bpcstop.pack(side=LEFT)      
        
        Label(self.frame, text='DBE', width=5).pack(side=LEFT)
        
        self.bpdbestart=Entry(self.frame,width=3)
        self.bpdbestart.insert(END,'0')
        self.bpdbestart.pack(side=LEFT)    
        
        Label(self.frame, text='–', width=3).pack(side=LEFT)

        self.bpdbestop=Entry(self.frame,width=3)
        self.bpdbestop.insert(END,'20')
        self.bpdbestop.pack(side=LEFT)    
        
        Label(self.frame,text='Plot class',width=10).pack(side=LEFT)
        
        self.bpclass=Entry(self.frame,width=10)
        self.bpclass.insert(END,'O2,N1')
        self.bpclass.pack(side=LEFT)
        
        Label(self.frame,text='Scaling',width=5).pack(side=LEFT)
        
        self.bpscale=Entry(self.frame,width=5)
        self.bpscale.insert(END,'1000')
        self.bpscale.pack(side=LEFT)
        
        self.bpshowc=IntVar()
        Checkbutton(self.frame,text='Disable class', variable=self.bpshowc).pack(side=LEFT)
        
        self.bpshows=IntVar()
        Checkbutton(self.frame,text='Disable name', variable=self.bpshows).pack(side=LEFT)    
        
        
                
class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.text_widget = Text(self)
        self.text_widget.pack()
        
        rawdataframe =RawDataFrame(self)
        bubbleplotframe=BubblePlotFrame(self)

        menubar = MenuBar(self,rawdataframe,bubbleplotframe)
        
        self.config(menu=menubar)
        
        

        
if __name__ == '__main__':
    app=App()
    app.title("FT-ICR MS Data Handler v0.1.2")
    app.mainloop()