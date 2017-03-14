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
ddf = ddf.loc[:,['EventId','Timestamp','Entity_Key', 'Entity_Value']]
eID_list = ddf['EventId'].dropna().values.astype('U')
final_eID_list = []

for ele in eID_list:
    if ele.isnumeric():
        final_eID_list.append(int(ele))
        
eID = np.array(final_eID_list, dtype='int64')


max_eID, min_eID = np.max(eID), np.min(eID)
eIDArr = np.array(np.arange(min_eID,max_eID+1),dtype='U')
 # the dataframe containing the eventid range from min to max
ix_df = pd.DataFrame(eIDArr, columns=('EventId',))


#split_ddf = dd.dataFrameSplit(ddf,)

#ddf1, ddf2, ddf3 = np.array_split(ddf,3)
#i=0
#df_arr = [ddf1, ddf2, ddf3]

def tblFromColVal(df, colname):
    uColVal = df[colname].drop_duplicates().dropna().values
    uColVal = ['app','noise_level_db','lat','lng','temp','level']
    colvalTbl = []

    for col in uColVal:
        x = df.loc[df[colname] == col]
        x.columns = ['EventId','Timestamp','Entity_Key', col]
        x = x.loc[:,['EventId','Timestamp',col]]
        x = x.drop_duplicates(subset=['Timestamp'])
        colvalTbl.append(x)
        
    return colvalTbl, uColVal
        
a, b = tblFromColVal(ddf,'Entity_Key')

i=0
for df in a:
    if i==0:
        fjoin = pd.merge(ix_df,df,how='left', on = 'EventId')
    else:
        fjoin = pd.merge(fjoin, df, how='left',on = 'EventId')
    i=i+1
    
split_fjoin = dd.dataFrameSplit(fjoin)
    
i=1
for csvfile in split_fjoin:    
    csvfile.to_csv("D:\\LearningContextIndividualFiles\\finalEventId_"+str(i)+".csv", sep=',',
              header=True, index=False)
    i=i+1


"""
a = sorted(a,key=len, reverse = True)
a = a[0:10]
i=0
for df in a:
    if i==0:
        fjoin = df
    else:
        fjoin = pd.merge(fjoin,df,how='left',on='Timestamp')
    i=i+1

#fjoin.to_csv("D:\\LearningContextIndividualFiles\\finaljoin.csv", sep=',',
#            header=True, index=False)
        
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
i=0
for df in a:
    if(i==0):
        fjoin = pd.merge(ix_df,df,how='left',on='EventId')
    else:
        fjoin = pd.merge(fjoin, df, how='left', on='EventId')
        
    i=i+1
    
fjoin.to_csv("D:\\LearningContextIndividualFiles\\finaljoin.csv", sep=',',
             header=True, index=False)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""             
"""
 Write the dataframes from the list to file
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def writedf(df_list, col_list, outfile):
    i = 0
    for df in df_list:
        fileName = col_list[i]+".csv"
        df.to_csv(path_or_buf=outfile + fileName, sep=',', na_rep='', float_format=None, 
                  columns=None, header=True, index=False, index_label=None, 
                  mode='w', encoding=None, compression=None, quoting=None, 
                  quotechar='"', line_terminator='\n', chunksize=None, 
                  tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')   
        i = i + 1
writedf(a, b, "D:\\LearningContextIndividualFiles\\")
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
for arr in df_arr:
    arr.to_csv("D:\\ddf"+str(i+1)+".csv",encoding='utf-8', index=False,
                         header=False)
    i = i+1

"""


    
    

