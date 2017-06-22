# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:38:07 2017

@author: Logesh Govindarajulu
"""
# Sklearn Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

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
base_classifiers = {'decisiontree': DecisionTreeClassifier(criterion='gini',max_depth=21),
                    'naivebayes': GaussianNB(),
                    'neuralnet' : MLPClassifier(max_iter=500),
                    'kneighbors' : KNeighborsClassifier(n_neighbors=2)}

for key, value in base_classifiers.items():
    folder_write = "D:\\lcif\\16032017-IndividualFiles\\skinDataset\\bagging\\"+ key + "\\"
    clf = BaggingClassifier(base_estimator=value,n_estimators=10,warm_start=False,max_samples=1.0)
    datasets = LoadData(trainfile,testfile,scalerfile)
    samp_var = SamplingMethod(samp_size,samp_method)
    load_n_sample = Load_N_Sample(datasets,samp_var)
    runML = RunML(load_n_sample,clf,folder_write)
    print("################################################################################")
    print(key+" Algorithm")
    print("################################################################################")
    runML.runML()
