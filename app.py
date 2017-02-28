# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:56:43 2017

@author: Logesh Govindarajulu
"""
import numpy as np
from datamanip.DataDeDuplication import Deduplication as dd

ddf = dd.dataDedup_csv("D:\\fulldataLearningContext.csv")
ddf1, ddf2, ddf3 = np.array_split(ddf,3)
df_arr = [ddf1, ddf2, ddf3]
i=0
for arr in df_arr:
    arr.to_csv("D:\\ddf"+str(i+1)+".csv",encoding='utf-8', index=False,
                         header=False)
    i = i+1
    
    
    