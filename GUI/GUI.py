from tkinter import *
from tkinter import ttk
import pandas as pd
from tkinter import filedialog
import os


def readClipboard():
    data = pd.read_clipboard()
    return data

def readExcel():
    excel_path=filedialog.askopenfilename(defaultextension='.xlsx', filetypes=(('Excel', '*.xlsx'), ('2003 Excel', '*.xls'), ('CSV', '*.csv'), ('All Files', '*.*')))
    if os.path.splitext(excel_path)[1] == '.xlsx' or 'xls':
        data = pd.read_excel(excel_path)
    elif os.path.splitext(excel_path)[1] == '.csv':
        data = pd.read_csv(excel_path)
    return data
    
def readFolder():
    folder_path=filedialog.askdirectory()
    return folder_path

root = Tk()

menu = Menu(root)
root.config(menu=menu)

fileMenu = Menu(menu)
menu.add_cascade(label='File', menu=fileMenu)
fileMenu.add_command(label='import from clipboard', command=readClipboard)
fileMenu.add_command(label='import from excel', command=readExcel)
fileMenu.add_command(label='import from folder',command=readFolder)



root.mainloop()
