
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:38:07 2017

@author: Logesh Govindarajulu
"""

from anapy.misc import ml_utility as mlu
from sklearn.neighbors import KNeighborsClassifier

train_dataset = mlu.load_dataset("D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\training_set1.dat")
test_dataset = mlu.load_dataset("D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\testing_set1.dat")
data_scaler = mlu.load_scaler("D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\standardscaler.dat")

ss, sm = mlu.samp_parameters([30,40,50,60,70,80,90,100],['random', 'systematic','stratified', 'cluster'])

clf = KNeighborsClassifier(n_neighbors=5)
folder_write = "D:\\lcif\\16032017-IndividualFiles\\yeastDataset\\kneighbors\\"
mlu.train_eval(clf,sm,train_dataset, test_dataset,ss,data_scaler, folder_write, write_to_file=True)
