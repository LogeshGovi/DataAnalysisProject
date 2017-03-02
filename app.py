# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:56:43 2017

@author: Logesh Govindarajulu
"""
import numpy as np
from datamanip.DataDeDuplication import Deduplication as dd
from datamanip.CentralValues import CentralValues as cv

ddf = dd.dataDedup_csv("D:\\fulldataLearningContext.csv")
split_ddf = dd.dataFrameSplit(ddf,)
"""
ddf1, ddf2, ddf3 = np.array_split(ddf,3)
i=0
df_arr = [ddf1, ddf2, ddf3]
for arr in df_arr:
    arr.to_csv("D:\\ddf"+str(i+1)+".csv",encoding='utf-8', index=False,
                         header=False)
    i = i+1
    
    
CenVal = cv((np.arange(1,101)))
cVal = CenVal.cenvalues()
print(cVal)
"""
