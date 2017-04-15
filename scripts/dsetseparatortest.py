# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 00:37:08 2017

@author: Logesh Govindarajulu
"""
import pandas as pd
from datamanip import datasetSeparator as ds

df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\prelimMLdataset.csv",
                  dtype='unicode')

dsep = ds.DataSeparator(df)
colheaders = dsep.displayCols()
fulldf = pd.DataFrame([])
fulldf = dsep.remCols([0,1,2])
a, b = dsep.sep_data_target(fulldf)
a = a.astype('float')
b = b.astype('float')