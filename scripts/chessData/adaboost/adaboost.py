
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:38:07 2017

@author: Logesh Govindarajulu
"""
# Sklearn Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

# Anapy imports
from anapy.misc.load_sample_ml import LoadData
from anapy.misc.load_sample_ml import SamplingMethod
from anapy.misc.load_sample_ml import Load_N_Sample
from anapy.misc.load_sample_ml import RunML

trainfile= "D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\training_set1.dat"
testfile="D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\testing_set1.dat"
scalerfile="D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\standardscaler.dat"
samp_size = [30,40,50,60,70,80,90,100]
samp_method = ['random', 'systematic','stratified', 'cluster']
base_classifiers = {'decisiontree': DecisionTreeClassifier(criterion='gini',max_depth=21),
                    'naivebayes': GaussianNB(),
                   }

for key, value in base_classifiers.items():
    folder_write = "D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\adaboost\\"+ key + "\\"
    clf = AdaBoostClassifier(base_estimator=value,n_estimators=50)
    datasets = LoadData(trainfile,testfile,scalerfile)
    samp_var = SamplingMethod(samp_size,samp_method)
    load_n_sample = Load_N_Sample(datasets,samp_var)
    runML = RunML(load_n_sample,clf,folder_write)
    print("################################################################################")
    print(key+" Algorithm")
    print("################################################################################")
    runML.runML()
