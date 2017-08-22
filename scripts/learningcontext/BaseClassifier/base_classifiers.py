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

# Anapy imports
from anapy.misc.load_sample_ml import LoadData
from anapy.misc.load_sample_ml import SamplingMethod
from anapy.misc.load_sample_ml import Load_N_Sample
from anapy.misc.load_sample_ml import RunML

"""
Load training data set, test data set, standard scaler
"""

trainfile= "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\training_set1.dat"
testfile="D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\testing_set1.dat"
scalerfile="D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\standardscaler.dat"
samp_size = [30,40,50,60,70,80,90,100]
samp_method = ['random', 'systematic','stratified', 'cluster']

"""
Classifier definition for the base learners:
1. Decision Tree
2. Naive Bayes
3. Neural Net
4. Kneighbors
"""

base_classifiers = {'decisiontree' : DecisionTreeClassifier(criterion="gini", max_depth=13),
                    'naivebayes': GaussianNB(),
                    'neuralnet': MLPClassifier(hidden_layer_sizes=(40,40,40,),max_iter=400,),
                    'kneighbors': KNeighborsClassifier(n_neighbors=5)
                    }


"""
Training, Testing and Compilation of the results
"""
for key, value in base_classifiers.items():
    folder_write = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\"+ key + "\\"
    clf = value
    datasets = LoadData(trainfile,testfile,scalerfile)
    samp_var = SamplingMethod(samp_size,samp_method)
    load_n_sample = Load_N_Sample(datasets,samp_var)
    runML = RunML(load_n_sample,clf,folder_write)
    print("################################################################################")
    print(key+" Algorithm")
    print("################################################################################")
    runML.runML()
