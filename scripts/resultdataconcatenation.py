# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:47:46 2017

@author: Logesh Govindarajulu
"""

import pandas as pd
import numpy as np
import os


sampling_method = ['random', 'systematic', 'stratified', 'cluster']
dataset_type = ['train', 'test']
sample_size = np.arange(30,101,10)
filetype = ".html"
base_clf =['decisiontree', 'kneighbors', 'naivebayes', 'neuralnet']
tree_ensemble = ['gradientboosting', 'extratrees', 'randomforest']
ensemble = ['bagging','adaboost']
adaboost_base = ['decisiontree','naivebayes']
bagging_base = ['decisiontree','kneighbors', 'naivebayes', 'neuralnet']

base_file_path = os.path.join("D:\\","lcif","16032017-IndividualFiles","TrainTestdataset")



sample_paths = []

for bclf in base_clf:
    for dst in dataset_type:
        for sm in sampling_method:
            paths = []
            for ssize in sample_size:
                fname = sm+"_"+dst+"_report_"+str(ssize)+filetype
                paths.append(os.path.join(base_file_path,bclf,fname))
            sample_paths.append(paths)
            del(paths)

 
dflist = []               
for idx, pthlist in enumerate(sample_paths):
    dataframes = []
    for ipth in pthlist:
        df = pd.read_html(ipth)
        dataframes.append(df[0])
    dflist.append(pd.concat(dataframes,axis=1))
    del(dataframes)
        
"""        
for idx, df in enumerate(dflist):
    df.to_html("dataframes\\"+str(idx+1)+".html")
"""                

