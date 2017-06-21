# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:38:07 2017

@author: Logesh Govindarajulu
"""

from anapy.misc import ml_utility as mlu
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

train_dataset = mlu.load_dataset("D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\training_set1.dat")
test_dataset = mlu.load_dataset("D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\testing_set1.dat")
data_scaler = mlu.load_scaler("D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\standardscaler.dat")

ss, sm = mlu.samp_parameters([30,40,50,60,70,80,90,100],['random', 'systematic','stratified', 'cluster'])

dtc = DecisionTreeClassifier(criterion="gini",max_depth=5)
clf = BaggingClassifier(dtc, n_estimators=10)
folder_write = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\bagging\\decisiontree\\"
mlu.train_eval(clf,sm,train_dataset, test_dataset,ss,data_scaler, folder_write, write_to_file=True)
