# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:25:42 2017

@author: samuel
"""

import base64
import itertools
import os
import threading
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icon import Icon
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# https://www.nist.gov/pml/handbook-basic-atomic-spectroscopic-data
atomic_mass = {'C': 12.000000, 'C13': 13.003355, 'H': 1.007825, 'N': 14.003074, 'O': 15.994915, 'S': 31.972070,
               'e': 0.0005485799, 'CH2': 14.01565, 'Na': 22.989767, 'Cl': 34.968852}
mass_tolerance = 0.0015 * atomic_mass['CH2'] / 14


class Compound:
    def __init__(self, c, h, n, o, s, na, cl, mode):
        self.c = c
        self.h = h
        self.o = o
        self.n = n
        self.s = s
        self.na = na
        self.cl = cl
        self.mode = mode
        self.ston = 0
        self.mw = self.mw()
        self.isomw = self.iso_mw()
        self.realh = self.realh()
        self.dbe = (2 * c + 2 + n - self.realh - self.na - self.cl) / 2
        self.memw = 0
        self.intensity = 0
        self.ppm = 0
        self.specie = self.specie()

    def mw(self):
        if self.mode == 1 or self.mode == 3:
            a = atomic_mass['C'] * self.c + atomic_mass['N'] * self.n + atomic_mass['S'] * self.s + atomic_mass[
                'O'] * self.o + atomic_mass['H'] * self.h + atomic_mass['Na'] * self.na + atomic_mass['Cl'] * self.cl - \
                atomic_mass['e']
        elif self.mode == 2:
            a = atomic_mass['C'] * self.c + atomic_mass['N'] * self.n + atomic_mass['S'] * self.s + atomic_mass[
                'O'] * self.o + atomic_mass['H'] * self.h + atomic_mass['Na'] * self.na + atomic_mass['Cl'] * self.cl + \
                atomic_mass['e']
        return a

    def iso_mw(self):
        if self.mode == 1 or self.mode == 3:
            a = atomic_mass['C'] * (self.c - 1) + atomic_mass['C13'] + atomic_mass['N'] * self.n + atomic_mass[
                'S'] * self.s + atomic_mass['O'] * self.o + atomic_mass['H'] * self.h + atomic_mass['Na'] * self.na + \
                atomic_mass['Cl'] * self.cl - atomic_mass['e']
        elif self.mode == 2:
            a = atomic_mass['C'] * (self.c - 1) + atomic_mass['C13'] + atomic_mass['N'] * self.n + atomic_mass[
                'S'] * self.s + atomic_mass['O'] * self.o + atomic_mass['H'] * self.h + atomic_mass['Na'] * self.na + \
                atomic_mass['Cl'] * self.cl + atomic_mass['e']
        return a

    def realh(self):
        if self.mode == 1 or self.mode == 3:
            b = self.h - 1
        if self.mode == 2:
            b = self.h + 1
        return b

    def specie(self):
        specie_n = ''
        specie_o = ''
        specie_s = ''
        specie_na = ''
        specie_cl = ''
        if self.n != 0:
            specie_n = 'N' + '%d' % self.n
        if self.o != 0:
            specie_o = 'O' + '%d' % self.o
        if self.s != 0:
            specie_s = 'S' + '%d' % self.s
        if self.na != 0:
            specie_na = 'Na' + '%d' % self.na
        if self.cl != 0:
            specie_cl = 'Cl' + '%d' % self.cl
        specie_all = specie_n + specie_o + specie_s + specie_na + specie_cl
        if specie_all == '':
            specie_all = 'CH'
        if self.dbe != int(self.dbe):
            specie_all = '*' + specie_all
        return specie_all


def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()


def isMolecule(a, mw_min):
    if a.mode == 1 or a.mode == 2:
        #        if a.n!=0 or a.o!=0 or a.s!=0:
        if 0.3 <= a.h / a.c <= 3.0:
            if a.o / a.c <= 3.0 and a.n / a.c <= 0.5:
                if a.realh <= 2 + 2 * a.c + a.n - a.na - a.cl:
                    if (a.h + a.n + a.na + a.cl) % 2 == 1:
                        if a.mw >= mw_min:
                            return True
    if a.mode == 3:
        if 0.3 <= a.h / a.c <= 3.0:
            if a.o / a.c <= 3.0 and a.n / a.c <= 0.5:
                if a.realh <= 2 + 2 * a.c + a.n - a.na - a.cl:
                    if a.mw >= mw_min:
                        return True


def excelSave(excelFile):
    path = filedialog.asksaveasfilename(defaultextension='.xlsx',
                                        filetypes=(('Excel', '*.xlsx'), ('2003 Excel', '*.xls'), ('All Files', '*.*')))
    if path is None:
        return
    else:
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
    excelFile.to_excel(writer, 'Sheet1',index=False)
    writer.save()


def readAllExcel(path):
    excelFilePath = []
    for root, dirs, files in os.walk(path):
        for excel in files:
            if os.path.splitext(excel)[1] == '.xlsx':
                excelFilePath.append(path + '/' + excel)
    return excelFilePath


class ParaDialog(Toplevel):

    def __init__(self):
        Toplevel.__init__(self)
        self.title('PARAMETERS')
        self.para = []
        self.setup_UI()

    def setup_UI(self):
        row1 = Frame(self)
        row1.pack()
        Label(row1, text='C', width=5).pack(side=LEFT)
        self.cacstart = IntVar()
        self.cacstop = IntVar()
        Entry(row1, textvariable=self.cacstart, width=5).pack(side=LEFT)
        Label(row1, text='–').pack(side=LEFT)
        Entry(row1, textvariable=self.cacstop, width=5).pack(side=LEFT)

        Label(row1, text='DBE', width=5).pack(side=LEFT)
        self.cadbe = IntVar()
        Entry(row1, textvariable=self.cadbe, width=5).pack(side=LEFT)

        Label(row1, text='Mode', width=5).pack(side=LEFT)
        self.camo = IntVar()
        Radiobutton(row1, text='+ESI', variable=self.camo, value=1).pack(side=LEFT)
        Radiobutton(row1, text='-ESI', variable=self.camo, value=2).pack(side=LEFT)

        row2 = Frame(self)
        row2.pack()
        Label(row2, text='N', width=5).pack(side=LEFT)
        self.can = IntVar()
        Entry(row2, textvariable=self.can, width=5).pack(side=LEFT)

        Label(row2, text='O', width=5).pack(side=LEFT)
        self.cao = IntVar()
        Entry(row2, textvariable=self.cao, width=5).pack(side=LEFT)

        Label(row2, text='S', width=5).pack(side=LEFT)
        self.cas = IntVar()
        Entry(row2, textvariable=self.cas, width=5).pack(side=LEFT)

        Label(row2, text='Na', width=5).pack(side=LEFT)
        self.cana = IntVar()
        Entry(row2, textvariable=self.cana, width=5).pack(side=LEFT)

        Label(row2, text='Cl', width=5).pack(side=LEFT)
        self.cacl = IntVar()
        Entry(row2, textvariable=self.cacl, width=5).pack(side=LEFT)

        row3 = Frame(self)
        row3.pack()
        Button(row3, text='Cancel', command=self.cancel).pack(side=RIGHT)

        Label(row3, text='\t').pack(side=RIGHT)

        Button(row3, text='OK', command=self.ok).pack(side=RIGHT)

    def ok(self):
        self.para = [self.cacstart.get(), self.cacstop.get(), self.cadbe.get(), self.camo.get(), self.can.get(),
                     self.cao.get(), self.cas.get(), self.cana.get(), self.cacl.get()]
        self.destroy()

    def cancel(self):
        self.para = None
        self.destroy()


class MenuBar(Menu):

    def __init__(self, parent, rawdataframe, bubbleplotframe):
        Menu.__init__(self, parent)

        fileMenu = Menu(self)

        self.bubbleplotframe = bubbleplotframe
        self.rawdataframe = rawdataframe

        self.add_cascade(label='Import', menu=fileMenu)
        fileMenu.add_command(label='From clipboard', command=lambda: thread_it(self.readClipboard))
        fileMenu.add_command(label='From file', command=lambda: thread_it(self.readExcel))
        fileMenu.add_command(label='From folder', command=lambda: thread_it(self.readFolder))

        self.text_widget = parent.text_widget

        self.data = pd.DataFrame().astype(float)
        self.folder_path = 0
        self.excelName = 0

        calMenu = Menu(self)
        self.add_cascade(label='Analysis', menu=calMenu)
        calMenu.add_command(label='Possible formulas', command=lambda: thread_it(self.processData))
        calMenu.add_command(label='Molecular weight calibration', command=lambda: thread_it(self.mwCa))
        calMenu.add_command(label='Class abundance from file', command=lambda: thread_it(self.calAbundance))
        calMenu.add_command(label='Class abundance from folder', command=lambda: thread_it(self.calAbundanceFile))
        calMenu.add_command(label='Class DBE abundance from file', command=lambda: thread_it(self.caldbeAbundance))
        calMenu.add_command(label='Class DBE abundance from folder',
                            command=lambda: thread_it(self.caldbeAbundanceFile))
        calMenu.add_command(label='Planar limits calculation', command=lambda: thread_it(self.calplanarlimits))
        calMenu.add_command(label='Customized Calculation 1', command=lambda: thread_it(self.cuscal1))
        calMenu.add_command(label='Merge table for PCA', command=lambda: thread_it(self.mergeTablePCA))
        calMenu.add_command(label='PCA', command=lambda: thread_it(self.pca))

        plotMenu = Menu(self)
        self.add_cascade(label='Plot', menu=plotMenu)
        plotMenu.add_command(label='Bar plot from file', command=lambda: thread_it(self.barplot))
        plotMenu.add_command(label='Bubble plots from file', command=lambda: thread_it(self.bubbleplotfile))
        plotMenu.add_command(label='Bubble plots from folder', command=lambda: thread_it(self.bubbleplot))

        aboutMenu = Menu(self)
        self.add_cascade(label='Help', menu=aboutMenu)
        aboutMenu.add_command(label='About', command=self.aboutMessage)

    def readClipboard(self):
        self.data = pd.read_clipboard()
        if not ('m/z' or 'I' or 'S/N') in self.data.columns:
            messagebox.showerror('Error', "Wrong data format!\nMUST include 'm/z', 'I', and 'S/N'")
        self.excelName = 'Clipboard'
        self.text_widget.delete('1.0', END)
        self.text_widget.insert(END, self.data)

    def readExcel(self):
        excel_path = filedialog.askopenfilename(defaultextension='.xlsx', filetypes=(
        ('Excel', '*.xlsx'), ('2003 Excel', '*.xls'), ('CSV', '*.csv'), ('All Files', '*.*')))
        if os.path.splitext(excel_path)[1] == '.xlsx' or os.path.splitext(excel_path)[1] == 'xls':
            self.data = pd.read_excel(excel_path)
        elif os.path.splitext(excel_path)[1] == '.csv':
            self.data = pd.read_csv(excel_path)
        excelName = os.path.splitext(excel_path)[0]
        self.excelName = excelName.split('\\')[-1]
        self.text_widget.delete('1.0', END)
        self.text_widget.insert(END, self.data)

    def readFolder(self):
        self.folder_path = filedialog.askdirectory()
        self.text_widget.delete('1.0', END)
        path = readAllExcel(self.folder_path)
        self.text_widget.insert(END, 'These are the Excels found in the path: \n')
        for paths in path:
            self.text_widget.insert(END, paths + '\n')

    def calAbundance(self):
        try:
            data = self.data
            if not 'normalized' in data.columns:
                data['normalized'] = data['intensity'] / data['intensity'].sum()
            species = data['class']
            species = species.drop_duplicates()
            abundance = pd.DataFrame().astype(float)
            for specie in species:
                data_specie = data[data['class'] == specie]
                abundance.loc[specie, self.excelName] = data_specie['normalized'].sum()
            self.text_widget.delete('1.0', END)
            self.text_widget.insert(END, abundance)
            excelSave(abundance)
        except:
            messagebox.showerror('Error', 'Please import data first!')

    def setPara(self):
        para = self.askPara()
        if para is None: return

        self.cacstart, self.cacstop, self.cadbe, self.camo, self.can, self.cao, self.cas, self.cana, self.cacl = para

    def askPara(self):
        inputDialog = ParaDialog()
        self.wait_window(inputDialog)
        return inputDialog.para

    def processData(self):
        if self.rawdataframe.modeEntry.get() == 1 or self.rawdataframe.modeEntry.get() == 2:
            self.processESIData()
        elif self.rawdataframe.modeEntry.get() == 3:
            self.processAPPIData()

    def processESIData(self):
        self.text_widget.insert(END, "Processing ESI data, please wait and do not close the window......")
        saveExcel = pd.DataFrame()
        for i in (
        'measured m/z', 'm/z', 'ppm', 'S/N', 'class', 'C', 'H', 'O', 'N', 'S', 'Na', 'Cl', 'DBE', 'intensity'):
            saveExcel.loc[0, i] = i
            i += i
        count = 0
        self.isodata = self.data
        self.data = self.data[self.data['S/N'] >= int(self.rawdataframe.snEntry.get())]
        for column in self.data:
            if column != 'm/z' and column != 'I' and column != 'S/N':
                del self.data[column]
        mw_max = self.data['m/z'].max()
        mw_min = self.data['m/z'].min()
        for N, O, S, Na, Cl in itertools.product(range(int(self.rawdataframe.nEntry.get()) + 1),
                                                 range(int(self.rawdataframe.oEntry.get()) + 1),
                                                 range(int(self.rawdataframe.sEntry.get()) + 1),
                                                 range(int(self.rawdataframe.naEntry.get()) + 1),
                                                 range(int(self.rawdataframe.clEntry.get()) + 1)):
            c_max = int((mw_max - atomic_mass['N'] * N - atomic_mass['O'] * O - atomic_mass['S'] * S - atomic_mass[
                'Na'] * Na - atomic_mass['Cl'] * Cl) / atomic_mass['C'])
            for C in range(1, c_max + 1):
                h_max = int((mw_max - atomic_mass['C'] * C - atomic_mass['N'] * N - atomic_mass['O'] * O - atomic_mass[
                    'S'] * S - atomic_mass['Na'] * Na - atomic_mass['Cl'] * Cl) / atomic_mass['H']) + 1
                for H in range(1, h_max + 1):
                    molecule = Compound(C, H, N, O, S, Na, Cl, self.rawdataframe.modeEntry.get())
                    if isMolecule(molecule, mw_min):
                        data_test = self.data[(self.data['m/z'] >= (molecule.mw - mass_tolerance)) & (
                                    self.data['m/z'] <= (molecule.mw + mass_tolerance))]
                        data_test_iso = self.isodata[(self.isodata['m/z'] >= (molecule.isomw - mass_tolerance)) & (
                                    self.isodata['m/z'] <= (molecule.isomw + mass_tolerance))]
                        if not data_test.empty and not data_test_iso.empty:
                            molecule.intensity = data_test['I'].max()
                            data_test = data_test[data_test['I'] == molecule.intensity]
                            data_test1 = data_test['m/z'].tolist()
                            molecule.memw = data_test1[0]
                            data_test2 = data_test['S/N'].tolist()
                            molecule.ston = data_test2[0]
                            molecule.ppm = abs(1000000 * (molecule.mw - molecule.memw) / molecule.mw)
                            if molecule.ppm <= float(self.rawdataframe.ppmEntry.get()):
                                stringTovar = {'measured m/z': molecule.memw, 'm/z': molecule.mw, 'ppm': molecule.ppm,
                                               'S/N': molecule.ston, 'class': molecule.specie, 'C': molecule.c,
                                               'H': molecule.realh, 'O': molecule.o, 'N': molecule.n, 'S': molecule.s,
                                               'Na': molecule.na, 'Cl': molecule.cl, 'DBE': molecule.dbe,
                                               'intensity': molecule.intensity}
                                for column in saveExcel:
                                    saveExcel.loc[count, column] = stringTovar[column]
                                count += 1
        self.text_widget.delete('1.0', END)
        self.text_widget.insert(END, saveExcel)
        excelSave(saveExcel)

    def processAPPIData(self):
        self.text_widget.insert(END, "Processing APPI data, please wait and do not close the window......")
        saveExcel = pd.DataFrame()
        for i in (
        'measured m/z', 'm/z', 'ppm', 'S/N', 'class', 'C', 'H', 'O', 'N', 'S', 'Na', 'Cl', 'DBE', 'intensity'):
            saveExcel.loc[0, i] = i
            i += i
        count = 0
        self.isodata = self.data
        self.data = self.data[self.data['S/N'] >= int(self.rawdataframe.snEntry.get())]
        for column in self.data:
            if column != 'm/z' and column != 'I' and column != 'S/N':
                del self.data[column]
        mw_max = self.data['m/z'].max()
        mw_min = self.data['m/z'].min()
        for N, O, S, Na, Cl in itertools.product(range(int(self.rawdataframe.nEntry.get()) + 1),
                                                 range(int(self.rawdataframe.oEntry.get()) + 1),
                                                 range(int(self.rawdataframe.sEntry.get()) + 1),
                                                 range(int(self.rawdataframe.naEntry.get()) + 1),
                                                 range(int(self.rawdataframe.clEntry.get()) + 1)):
            c_max = int((mw_max - atomic_mass['N'] * N - atomic_mass['O'] * O - atomic_mass['S'] * S - atomic_mass[
                'Na'] * Na - atomic_mass['Cl'] * Cl) / atomic_mass['C'])
            for C in range(1, c_max + 1):
                h_max = int((mw_max - atomic_mass['C'] * C - atomic_mass['N'] * N - atomic_mass['O'] * O - atomic_mass[
                    'S'] * S - atomic_mass['Na'] * Na - atomic_mass['Cl'] * Cl) / atomic_mass['H']) + 1
                for H in range(1, h_max + 1):
                    molecule = Compound(C, H, N, O, S, 0, self.rawdataframe.modeEntry.get())
                    if isMolecule(molecule, mw_min):
                        data_test = self.data[(self.data['m/z'] >= (molecule.mw - mass_tolerance)) & (
                                    self.data['m/z'] <= (molecule.mw + mass_tolerance))]
                        data_test_iso = self.isodata[(self.isodata['m/z'] >= (molecule.isomw - mass_tolerance)) & (
                                    self.isodata['m/z'] <= (molecule.isomw + mass_tolerance))]
                        if not data_test.empty and not data_test_iso.empty:
                            molecule.intensity = data_test['I'].max()
                            data_test = data_test[data_test['I'] == molecule.intensity]
                            data_test1 = data_test['m/z'].tolist()
                            molecule.memw = data_test1[0]
                            data_test2 = data_test['S/N'].tolist()
                            molecule.ston = data_test2[0]
                            molecule.ppm = abs(1000000 * (molecule.mw - molecule.memw) / molecule.mw)
                            if molecule.ppm <= float(self.rawdataframe.ppmEntry.get()):
                                stringTovar = {'measured m/z': molecule.memw, 'm/z': molecule.mw, 'ppm': molecule.ppm,
                                               'S/N': molecule.ston, 'class': molecule.specie, 'C': molecule.c,
                                               'H': molecule.realh, 'O': molecule.o, 'N': molecule.n, 'S': molecule.s,
                                               'Na': molecule.na, 'Cl': molecule.cl, 'DBE': molecule.dbe,
                                               'intensity': molecule.intensity}
                                for column in saveExcel:
                                    saveExcel.loc[count, column] = stringTovar[column]
                                count += 1
        self.text_widget.delete('1.0', END)
        self.text_widget.insert(END, saveExcel)
        excelSave(saveExcel)

    def mwCa(self):
        self.setPara()
        capath = filedialog.asksaveasfilename(defaultextension='ref')
        caref = open(capath, 'a')
        if self.camo == 1:
            for cac in range(self.cacstart, self.cacstop):
                camolecule = Compound(cac, 2 * cac + 3 + self.can - 2 * self.cadbe, self.can, self.cao, self.cas,
                                      self.cana, self.cacl, 1)
                caformula = 'C' + str(cac) + 'H' + str(camolecule.h)
                if not self.can == 0:
                    caformula = caformula + 'N' + str(camolecule.n)
                if not self.cao == 0:
                    caformula = caformula + 'O' + str(camolecule.o)
                if not self.cas == 0:
                    caformula = caformula + 'S' + str(camolecule.s)
                if not self.cana == 0:
                    caformula = caformula + 'Na' + str(camolecule.na)
                if not self.cacl == 0:
                    caformula = caformula + 'Cl' + str(camolecule.cl)
                caref.write(caformula + ' ' + str(camolecule.mw) + ' ' + '1+')
                caref.write('\n')

        if self.camo == 2:
            for cac in range(self.cacstart, self.cacstop):
                camolecule = Compound(cac, 2 * cac + 1 + self.can - 2 * self.cadbe, self.can, self.cao, self.cas,
                                      self.cana, self.cacl, 2)
                caformula = 'C' + str(cac) + 'H' + str(camolecule.h)
                if not self.can == 0:
                    caformula = caformula + 'N' + str(camolecule.n)
                if not self.cao == 0:
                    caformula = caformula + 'O' + str(camolecule.o)
                if not self.cas == 0:
                    caformula = caformula + 'S' + str(camolecule.s)
                if not self.cana == 0:
                    caformula = caformula + 'Na' + str(camolecule.na)
                if not self.cacl == 0:
                    caformula = caformula + 'Cl' + str(camolecule.cl)
                caref.write(caformula + ' ' + str(camolecule.mw) + ' ' + '1-')
                caref.write('\n')
        caref.close()

    def calAbundanceFile(self):
        try:
            excelFile = readAllExcel(self.folder_path)
            abundance = pd.DataFrame().astype(float)
            for excel in excelFile:
                self.excelName = os.path.split(excel)[1]
                self.excelName = os.path.splitext(self.excelName)[0]
                self.data = pd.read_excel(excel)
                data = self.data
                if not 'normalized' in data.columns:
                    data['normalized'] = data['intensity'] / data['intensity'].sum()
                species = data['class']
                species = species.drop_duplicates()
                for specie in species:
                    data_specie = data[data['class'] == specie]
                    abundance.loc[specie, self.excelName] = data_specie['normalized'].sum()
                self.text_widget.delete('1.0', END)
                self.text_widget.insert(END, abundance)
            excelSave(abundance)
        except:
            messagebox.showerror('Error', 'Please import data first!')

    def caldbeAbundance(self):
        try:
            data = self.data
            if not 'normalized' in data.columns:
                data['normalized'] = data['intensity'] / data['intensity'].sum()
            species = data['class']
            species = species.drop_duplicates()
            abundance = pd.DataFrame().astype(float)
            dbe = 0
            for specie in species:
                data_specie = data[data['class'] == specie]
                for dbe in range(0, 20):
                    data_dbe = data_specie[data_specie['DBE'] == dbe]
                    abundance.loc[specie, dbe] = data_dbe['normalized'].sum()
            self.text_widget.delete('1.0', END)
            self.text_widget.insert(END, abundance)
            excelSave(abundance)
        except:
            messagebox.showerror('Error', 'Please import data first!')

    def caldbeAbundanceFile(self):
        try:
            excelFile = readAllExcel(self.folder_path)
            for excel in excelFile:
                self.excelName = excel
                self.data = pd.read_excel(excel)
                self.caldbeAbundance()
        except:
            messagebox.showerror('Error', 'Please import data first!')

    def mergeTablePCA(self):
        self.text_widget.insert(END, "Merging Tables for PCA, please wait and do not close the window......")
        excelFile = readAllExcel(self.folder_path)
        pcatable = pd.DataFrame()
        basket = pd.DataFrame()
        mass = set()
        for excel in excelFile:
            raw = pd.read_excel(excel)
            raw = raw.dropna(axis=1, how='all')
            raw = raw.dropna()
            raw['formula'] = 'C' + raw['C'].astype(int).astype(str) + '-H' + raw['H'].astype(int).astype(str) + '-O' + raw[
                'O'].astype(int).astype(str) + '-N' + raw['N'].astype(int).astype(str)
            raw['m/z'] = raw['m/z'].astype(str) + ',' + raw['formula']
            excel=os.path.basename(excel)
            excel = excel.replace('.xlsx', '')
            excel = excel.replace('-', '_')
            for column in raw:
                if column != 'm/z' and column != 'intensity':
                    del raw[column]
            raw['normalized'] = raw['intensity'] / raw['intensity'].sum()
            del raw['intensity']
            raw = raw.rename(columns={'m/z': 'mass%s' % excel, 'normalized': excel, 'formula': 'formula%s' % excel})
            pcatable = pd.concat([pcatable, raw], axis=1, sort=False)
        for i in excelFile:
            i = os.path.basename(i)
            i = i.replace('.xlsx', '')
            i = i.replace('-', '_')
            locals()['pcatable' + i] = pcatable[['mass' + i, i]].dropna()
            mass.update(pcatable['mass' + i])
        mass = {x for x in mass if pd.notna(x)}
        for m in excelFile:
            m = os.path.basename(m)
            m = m.replace('.xlsx', '')
            m = m.replace('-', '_')
            locals()['masse' + m] = mass - set(pcatable['mass' + m])
            locals()['masse' + m] = pd.DataFrame(locals()['masse' + m], columns=['mass' + m])
            locals()['pcatable' + m] = pd.concat([locals()['pcatable' + m], locals()['masse' + m]], ignore_index=True,
                                                 sort=False).fillna(0)
            locals()['pcatable' + m] = locals()['pcatable' + m].sort_values(by=['mass' + m])
            locals()['pcatable' + m] = locals()['pcatable' + m].reset_index(drop=True)
            basket = pd.concat([basket, locals()['pcatable' + m]], axis=1, sort=False)
        basket = basket.rename(columns={('mass'+os.path.basename(excelFile[0]).replace('.xlsx', '').replace('-', '_')):"mtoz"})
        for column in basket:
            if 'mass' in column:
                del basket[column]
        basket[['mtoz', 'formula']] = basket['mtoz'].str.split(',', expand=True)
        basket['mtoz'] = basket['mtoz'].astype(float)
        basket = basket.sort_values(by=['mtoz'])
        self.text_widget.delete('1.0', END)
        self.text_widget.insert(END, basket)
        excelSave(basket)

    def calplanarlimits(self):
        data = self.data
        species = data['class']
        species = species.drop_duplicates()
        cn = 0
        dbe = 0
        planar = pd.DataFrame().astype(float)
        for specie in species:
            data_specie = data[data['class'] == specie]
            for cn in range(10, 90):
                data_cn = data_specie[data_specie['C'] == cn]
                planar.loc[specie, cn] = data_cn['DBE'].max()
        self.text_widget.delete('1.0', END)
        self.text_widget.insert(END, planar)
        excelSave(planar)

    # def pcatable(self):
    #     mz = set()
    #     excelFile = readAllExcel(self.folder_path)
    #     for excel in excelFile:
    #         data = pd.read_excel(excel)
    #         data = data[data.ppm < 1.2]
    #         mz_tmp = set(data['m/z'].unique())
    #         mz.update(mz_tmp)
    #     pca = pd.DataFrame().astype(float)
    #     for excel in excelFile:
    #         data = pd.read_excel(excel)
    #         excelName = excel
    #         data = data[data.ppm < 1.2]
    #         for mztmp in mz:
    #             data_mz = data[data['m/z'] == mztmp]
    #             if not data_mz.empty:
    #                 data_test3 = data_mz['class'].tolist()
    #                 pca.loc[mztmp, 'class'] = data_test3[0]
    #                 data_test2 = data_mz['RA'].tolist()
    #                 pca.loc[mztmp, excelName] = data_test2[0]
    #             else:
    #                 pca.loc[mztmp, excelName] = 0
    #     self.text_widget.delete('1.0', END)
    #     self.text_widget.insert(END, pca)
    #     excelSave(pca)

    def pca(self):
        data = self.data
        data = StandardScaler().fit_transform(data)
        data_pca = PCA(n_components=2)
        pca = data_pca.fit_transform(data)
        pca_excel = pd.DataFrame(pca, columns=['X1', 'X2'])
        excelSave(pca_excel)
        print(data_pca.explained_variance_)
        print(data_pca.explained_variance_ratio_)

    def cuscal1(self):
        os.chdir(self.folder_path)
        excelFile = readAllExcel(self.folder_path)
        species = self.bubbleplotframe.bpclass.get()
        species = species.split(',')
        cuscal1execel = pd.DataFrame().astype(float)
        for excel in excelFile:
            data = pd.read_excel(excel)
            data = data[data['DBE'] > 0]
            excelName = os.path.split(excel)[1]
            excelName = os.path.splitext(excelName)[0]
            data['intensity'] = data['intensity'].astype(float)
            data['H/C'] = data['H'] / data['C']
            for specie in species:
                data_specie = data[data['class'] == specie]
                data_specie = data_specie.reset_index()
                min_max_scaler = MinMaxScaler()
                data_specie['normalized'] = min_max_scaler.fit_transform(data_specie['intensity'].values.reshape(-1, 1))
                cuscal1execel[excelName + 'm/z'] = data_specie['m/z']
                cuscal1execel[excelName + 'normalized'] = data_specie['normalized']
                cuscal1execel[excelName + 'H/C'] = data_specie['H/C']
        excelSave(cuscal1execel)

    def barplot(self):
        try:
            data = self.data
            plt.figure(figsize=(15, 10))
            plt.bar(data.index, data.iloc[:, 0], align='center', alpha=0.5)
            plt.show()
        except:
            messagebox.showerror('Error', 'Please import data first!')

    def bubbleplotfile(self):
        species = self.bubbleplotframe.bpclass.get()
        excelName = os.path.split(self.excelName)[1]
        species = species.split(',')
        data = self.data
        data = data[data['DBE'] > 0]
        data['intensity'] = data['intensity'].astype(float)
        path = filedialog.askdirectory()
        for specie in species:
            data_specie = data[data['class'] == specie]
            sum = data_specie['intensity'].sum()
            data_specie['normalized'] = data_specie['intensity'] / sum
            plt.figure(figsize=(6, 5))
            font = {'family': 'arial',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 20,
                    }
            plt.axis([int(self.bubbleplotframe.bpcstart.get()), int(self.bubbleplotframe.bpcstop.get()),
                      int(self.bubbleplotframe.bpdbestart.get()), int(self.bubbleplotframe.bpdbestop.get())])
            plt.xlabel("Carbon Number", fontdict=font)
            plt.ylabel("DBE", fontdict=font)
            plt.xticks(fontsize=16, fontname='arial')
            plt.yticks(
                np.arange(int(self.bubbleplotframe.bpdbestart.get()), int(self.bubbleplotframe.bpdbestop.get()) + 1, 2),
                fontsize=16, fontname='arial')
            if self.bubbleplotframe.bpshowc.get() == 0:
                plt.text(int(self.bubbleplotframe.bpcstart.get()) + 1, int(self.bubbleplotframe.bpdbestop.get()) - 3,
                         s=specie, fontdict=font)
            if self.bubbleplotframe.bpshows.get() == 0:
                plt.text(int(self.bubbleplotframe.bpcstop.get()) - 5, int(self.bubbleplotframe.bpdbestop.get()) - 3,
                         s=excelName, fontdict=font)
            plt.scatter(data_specie['C'], data_specie['DBE'],
                        s=float(self.bubbleplotframe.bpscale.get()) * data_specie['normalized'], edgecolors='black',
                        linewidth=0.1)
            filename = specie + '.png'
            plt.savefig(os.path.join(path, filename), dpi=1000)
        messagebox.showinfo("Complete!", "All plots are stored successfully!")

    #    def hexbinplotfile(self):
    #        excelName=os.path.split(self.excelName)[1]
    #        data=self.data
    #        plt.hexbin(data['measured m/z'],data['H/C'],data['intensity'],gridsize=10000)
    #        plt.show()

    def bubbleplot(self):
        os.chdir(self.folder_path)
        excelFile = readAllExcel(self.folder_path)
        species = self.bubbleplotframe.bpclass.get()
        species = species.split(',')
        for specie in species:
            if os.path.exists(specie) == False:
                os.makedirs(specie)
        for excel in excelFile:
            data = pd.read_excel(excel)
            data = data[data['DBE'] > 0]
            excelName = os.path.split(excel)[1]
            excelName = os.path.splitext(excelName)[0]
            data['intensity'] = data['intensity'].astype(float)
            for specie in species:
                data_specie = data[data['class'] == specie]
                sum = data_specie['intensity'].sum()
                data_specie['normalized'] = data_specie['intensity'] / sum
                plt.figure(figsize=(6, 5))
                font = {'family': 'arial',
                        'color': 'black',
                        'weight': 'normal',
                        'size': 20,
                        }
                plt.axis([int(self.bubbleplotframe.bpcstart.get()), int(self.bubbleplotframe.bpcstop.get()),
                          int(self.bubbleplotframe.bpdbestart.get()), int(self.bubbleplotframe.bpdbestop.get())])
                plt.xlabel("Carbon Number", fontdict=font)
                plt.ylabel("DBE", fontdict=font)
                plt.xticks(fontsize=16, fontname='arial')
                plt.yticks(
                    np.arange(int(self.bubbleplotframe.bpdbestart.get()), int(self.bubbleplotframe.bpdbestop.get()) + 1,
                              2), fontsize=16, fontname='arial')
                if self.bubbleplotframe.bpshowc.get() == 0:
                    plt.text(int(self.bubbleplotframe.bpcstart.get()) + 1,
                             int(self.bubbleplotframe.bpdbestop.get()) - 3, s=specie, fontdict=font)
                if self.bubbleplotframe.bpshows.get() == 0:
                    plt.text(int(self.bubbleplotframe.bpcstop.get()) - 5, int(self.bubbleplotframe.bpdbestop.get()) - 3,
                             s=excelName, fontdict=font)
                plt.scatter(data_specie['C'], data_specie['DBE'],
                            s=float(self.bubbleplotframe.bpscale.get()) * data_specie['normalized'], edgecolors='black',
                            linewidth=0.1)
                path = self.folder_path + "\\" + specie
                filename = excelName + '.png'
                plt.savefig(os.path.join(path, filename), dpi=600)
        messagebox.showinfo("Complete!", "All plots are stored in the same folder with excels")

    def aboutMessage(self):
        messagebox.showinfo(title='About',
                            message='FT–ICR MS Data Handler\nLicensed under the terms of the Apache License 2.0\n\nDeveloped and maintained by Weimin Liu\n\nFor bug reports and feature requests, please open tickets\n\nSpecial thanks to the following contributors:\nDr. Bin Jiang (VB codes for raw data processing)\nDr. Yahe Zhang & Dr. Linzhou Zhang (VB codes for molecular weight calibration)\nDinosoft Labs (''Atom'' icon)\nLiqian Huang (Love and support)\n\nCoding for a better wolrd!')


class RawDataFrame:

    def __init__(self, parent):
        self.frame = Frame(parent)
        self.frame.pack()

        Label(self.frame, text='RAW DATA', width=8).pack(side=LEFT)
        Label(self.frame, text='  ', width=2).pack(side=LEFT)

        Label(self.frame, text='S/N', width=3).pack(side=LEFT)
        self.snEntry = Entry(self.frame, width=3)
        self.snEntry.insert(END, '6')
        self.snEntry.pack(side=LEFT)

        Label(self.frame, text='Error (ppm)', width=10).pack(side=LEFT)
        self.ppmEntry = Entry(self.frame, width=3)
        self.ppmEntry.insert(END, '1.2')
        self.ppmEntry.pack(side=LEFT)

        Label(self.frame, text='N', width=3).pack(side=LEFT)
        self.nEntry = Entry(self.frame, width=3)
        self.nEntry.insert(END, '5')
        self.nEntry.pack(side=LEFT)

        Label(self.frame, text='O', width=3).pack(side=LEFT)
        self.oEntry = Entry(self.frame, width=3)
        self.oEntry.insert(END, '5')
        self.oEntry.pack(side=LEFT)

        Label(self.frame, text='S', width=3).pack(side=LEFT)
        self.sEntry = Entry(self.frame, width=3)
        self.sEntry.insert(END, '5')
        self.sEntry.pack(side=LEFT)

        Label(self.frame, text='Na', width=3).pack(side=LEFT)
        self.naEntry = Entry(self.frame, width=3)
        self.naEntry.insert(END, '0')
        self.naEntry.pack(side=LEFT)

        Label(self.frame, text='Cl', width=3).pack(side=LEFT)
        self.clEntry = Entry(self.frame, width=3)
        self.clEntry.insert(END, '0')
        self.clEntry.pack(side=LEFT)

        Label(self.frame, text='Source', width=5).pack(side=LEFT)

        self.modeEntry = IntVar()

        Radiobutton(self.frame, text='+ESI', variable=self.modeEntry, value=1).pack(side=LEFT)
        Radiobutton(self.frame, text='-ESI', variable=self.modeEntry, value=2).pack(side=LEFT)
        Radiobutton(self.frame, text='APPI (Beta)', variable=self.modeEntry, value=3).pack(side=LEFT)


#        Radiobutton(self.frame,text='-APPI',variable=self.modeEntry,value=4).pack(side=LEFT)


class BubblePlotFrame:

    def __init__(self, parent):
        self.frame = Frame(parent)
        self.frame.pack()

        Label(self.frame, text='BUBBLE PLOT', width=12).pack(side=LEFT)

        Label(self.frame, text='C', width=3).pack(side=LEFT)

        self.bpcstart = Entry(self.frame, width=3)
        self.bpcstart.insert(END, '10')
        self.bpcstart.pack(side=LEFT)

        Label(self.frame, text='–', width=3).pack(side=LEFT)

        self.bpcstop = Entry(self.frame, width=3)
        self.bpcstop.insert(END, '50')
        self.bpcstop.pack(side=LEFT)

        Label(self.frame, text='DBE', width=5).pack(side=LEFT)

        self.bpdbestart = Entry(self.frame, width=3)
        self.bpdbestart.insert(END, '0')
        self.bpdbestart.pack(side=LEFT)

        Label(self.frame, text='–', width=3).pack(side=LEFT)

        self.bpdbestop = Entry(self.frame, width=3)
        self.bpdbestop.insert(END, '20')
        self.bpdbestop.pack(side=LEFT)

        Label(self.frame, text='Class', width=4).pack(side=LEFT)

        self.bpclass = Entry(self.frame, width=10)
        self.bpclass.insert(END, 'O2,N1')
        self.bpclass.pack(side=LEFT)

        Label(self.frame, text='Scaling', width=5).pack(side=LEFT)

        self.bpscale = Entry(self.frame, width=5)
        self.bpscale.insert(END, '1000')
        self.bpscale.pack(side=LEFT)

        self.bpshowc = IntVar()
        Checkbutton(self.frame, text='Disable class', variable=self.bpshowc).pack(side=LEFT)

        self.bpshows = IntVar()
        Checkbutton(self.frame, text='Disable name', variable=self.bpshows).pack(side=LEFT)


class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.text_widget = Text(self, width=100)
        self.text_widget.pack(fill='x', expand=1)

        rawdataframe = RawDataFrame(self)
        bubbleplotframe = BubblePlotFrame(self)

        menubar = MenuBar(self, rawdataframe, bubbleplotframe)

        self.config(menu=menubar)


if __name__ == '__main__':
    app = App()
    app.title("POLARIS v0.1.3")
    with open('tmp.ico', 'wb') as tmp:
        tmp.write(base64.b64decode(Icon().img))
    app.iconbitmap('tmp.ico')
    os.remove('tmp.ico')
    app.mainloop()
