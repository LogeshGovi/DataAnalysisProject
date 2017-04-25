# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:27:20 2017

@author: Logesh Govindarajulu
"""
import pandas as pd
from anapy.datamanip import datasetSeparator as ds
from anapy.mlops import visualization as vs

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\prelimMLdatasetFull.csv",
                  dtype='unicode')

dsep = ds.DataSeparator(df)
colheaders = dsep.displayCols()
fulldf = pd.DataFrame([])
fulldf = dsep.remCols([0,1,2,7,6,5])
a, b = dsep.sep_data_target(fulldf)
a = a.astype('float')
b = b.astype('float')
b= np.ravel(b)

vdata = vs.Visualization(a,b)
t_data = vdata.dim_red('PCA')
plots = vdata.visualizedata(t_data)

