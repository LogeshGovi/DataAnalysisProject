# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:27:20 2017

@author: Logesh Govindarajulu
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from anapy.datamanip import datasetSeparator as ds
from anapy.mlops import visualization as vs
import pickle
import numpy as np


from matplotlib.colors import ListedColormap


############################################
#   Learning Context Data                  #
############################################
df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\prelimMLdatasetFull.csv",
                  dtype='unicode')
datalabel0 = df.ix[:,1].unique().astype(str)
dsep = ds.DataSeparator(df)
colheaders = dsep.displayCols()
fulldf = pd.DataFrame([])
fulldf = dsep.remCols([0,1,2,3,13])
a, b = dsep.sep_data_target(fulldf)
a = a.astype('float')
a_std = StandardScaler().fit_transform(a)
b = b.astype('float')
b= np.ravel(b)
############################################



############################################
#   yeast Data                             #
############################################
yeastds = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\yeastDataset\\yeast.csv")
dsep2 = ds.DataSeparator(yeastds)
datalabel1 = yeastds.ix[:,1].unique().astype(str)
colheaders2 = dsep2.displayCols()
fulldf1 = pd.DataFrame()
fulldf1 = dsep2.remCols([0,1])
a1, b1 = dsep2.sep_data_target(fulldf1)
a1= a1.astype('float')
a1_std = StandardScaler().fit_transform(a1)
b1 = b1.astype('float')
############################################

"""
finalmlset = []
finalmlset.append(a1_std)
finalmlset.append(b1)
PIK = "D:\\lcif\\16032017-IndividualFiles\\yeastpickle.dat"
with open(PIK,'wb') as f:
    pickle.dump(finalmlset, f)
"""

"""
colours = [
            "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
            "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
            "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
            "#61615A"
          ]
"""

colours = [
            "#808080", "#FF0000","#FFFF00","#008000","#0000FF","#800080","#808000","#C1FFC1","#004D43"
          ]
                  
color_list = ListedColormap(colours)

def visualizedata(data, target, datalabel):
    vdata = vs.Visualization(data,target)
    dimred_algorithms = ['FactorAnalysis', 'FastICA', 'PCA', 'LinearDiscriminantAnalysis']
    for algos in  dimred_algorithms:
        t_data, dimred_method = vdata.dim_red(algos,n_dim=3)
        t2_data, dimred_method2 = vdata.dim_red(algos,n_dim=2)
        vdata.visualizedata(t2_data,dimred_method2,colmap=color_list,collabel=datalabel)
        vdata.visualizedata3d(t_data, dimred_method,colmap=color_list,collabel=datalabel)

visualizedata(a1,b1, datalabel1)
visualizedata(a1_std,b1,datalabel1)