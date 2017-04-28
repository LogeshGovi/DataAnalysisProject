# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:27:20 2017

@author: Logesh Govindarajulu
"""
import pandas as pd
from anapy.datamanip import datasetSeparator as ds
from anapy.mlops import visualization as vs

import numpy as np


from matplotlib.colors import ListedColormap

df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\prelimMLdatasetFull.csv",
                  dtype='unicode')

datalabel = df.ix[:,0].unique().astype(str)


dsep = ds.DataSeparator(df)
colheaders = dsep.displayCols()
fulldf = pd.DataFrame([])
fulldf = dsep.remCols([0,1,2])
a, b = dsep.sep_data_target(fulldf)
a = a.astype('float')
b = b.astype('float')
b= np.ravel(b)

colours = [
            "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
            "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
            "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
            "#61615A"
          ]
                  
color_list = ListedColormap(colours)

vdata = vs.Visualization(a,b)
dimred_algorithms = ['FactorAnalysis', 'FastICA', 'PCA', 'LinearDiscriminantAnalysis']
for algos in  dimred_algorithms:
    t_data, dimred_method = vdata.dim_red(algos,n_dim=3)
    t2_data, dimred_method2 = vdata.dim_red(algos,n_dim=2)
    plots = vdata.visualizedata(t2_data,dimred_method2,colmap=color_list,collabel=datalabel)
    plots2 = vdata.visualizedata3d(t_data, dimred_method,colmap=color_list,collabel=datalabel)


