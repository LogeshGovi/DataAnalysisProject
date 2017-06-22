
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:38:07 2017

@author: Logesh Govindarajulu
"""
# Sklearn Imports
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Anapy imports
from anapy.misc.load_sample_ml import LoadData
from anapy.misc.load_sample_ml import SamplingMethod
from anapy.misc.load_sample_ml import Load_N_Sample
from anapy.misc.load_sample_ml import RunML

trainfile= "D:\\lcif\\16032017-IndividualFiles\\skinDataset\\training_set1.dat"
testfile="D:\\lcif\\16032017-IndividualFiles\\skinDataset\\testing_set1.dat"
scalerfile="D:\\lcif\\16032017-IndividualFiles\\skinDataset\\standardscaler.dat"
samp_size = [30,40,50,60,70,80,90,100]
samp_method = ['random', 'systematic','stratified', 'cluster']
base_classifiers = {'gradientboosting': GradientBoostingClassifier(loss='deviance',learning_rate=0.1,
                                        n_estimators=25,max_depth=10,warm_start=False),
                    'randomforest': RandomForestClassifier(n_estimators=10, criterion='gini',max_depth=21,warm_start=False),
                    'extratrees' : ExtraTreesClassifier(n_estimators=10,criterion='gini',max_depth=21,warm_start=False),
                   }

for key, value in base_classifiers.items():
    folder_write = "D:\\lcif\\16032017-IndividualFiles\\skinDataset\\"+ key + "\\"
    clf = value
    datasets = LoadData(trainfile,testfile,scalerfile)
    samp_var = SamplingMethod(samp_size,samp_method)
    load_n_sample = Load_N_Sample(datasets,samp_var)
    runML = RunML(load_n_sample,clf,folder_write)
    print("################################################################################")
    print(key+" Algorithm")
    print("################################################################################")
    runML.runML()
