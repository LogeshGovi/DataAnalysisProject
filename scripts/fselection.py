# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 00:37:08 2017

@author: Logesh Govindarajulu
"""
import pandas as pd
from anapy.datamanip import datasetSeparator as ds

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\prelimMLdatasetFull.csv",
                  dtype='unicode')

dsep = ds.DataSeparator(df)
colheaders = dsep.displayCols()
fulldf = pd.DataFrame([])
fulldf = dsep.remCols([0,1,2])
a, b = dsep.sep_data_target(fulldf)
a = a.astype('float')
b = b.astype('float')
b= np.ravel(b)


