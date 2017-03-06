# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:56:43 2017

@author: Logesh Govindarajulu
"""
import numpy as np
from datamanip.DataOps import Deduplication as dd
from datamanip.CentralValues import CentralValues as cv
import pandas as pd

#ddf = dd.dataDedup_csv("D:\\fulldataLearningContext.csv")
ddf = pd.read_csv("D:\\fulldataLearningContext.csv", dtype='unicode')
#split_ddf = dd.dataFrameSplit(ddf,)

#ddf1, ddf2, ddf3 = np.array_split(ddf,3)
#i=0
#df_arr = [ddf1, ddf2, ddf3]


def tblFromColVal(df, colname):
    uColVal = df[colname].drop_duplicates().dropna().values
    colvalTbl = []

    for col in uColVal:
        x = df.loc[df[colname] == col]
        colvalTbl.append(x)
        
    return colvalTbl, uColVal
        
a, b = tblFromColVal(ddf,'Entity_Key')
    
"""
 Write the dataframes from the list to file
"""
def writedf(df_list, col_list, outfile):
    i = 0
    for df in df_list:
        fileName = col_list[i]+".csv"
        df.to_csv(path_or_buf=outfile + fileName, sep=',', na_rep='', float_format=None, 
                  columns=None, header=False, index=False, index_label=None, 
                  mode='w', encoding=None, compression=None, quoting=None, 
                  quotechar='"', line_terminator='\n', chunksize=None, 
                  tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')   
        i = i + 1


writedf(a, b, "D:\\LearningContextIndividualFiles\\")
"""
for arr in df_arr:
    arr.to_csv("D:\\ddf"+str(i+1)+".csv",encoding='utf-8', index=False,
                         header=False)
    i = i+1

"""
    
    

