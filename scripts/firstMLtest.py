# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:49:01 2017

@author: Logesh Govindarajulu
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier

from sklearn import decomposition

# Decision Tree learning 
df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\prelimMLdataset.csv",
                  dtype='unicode')

cols = [0,1,-1]
rows =[0]
df.drop(df.columns[cols],axis=1,inplace=True)
targetdf = df[[-1]]
datadf = df[[0,1,2,3,4,5,6,7]]
data = datadf.as_matrix()
target = targetdf.as_matrix()
data_train, data_test, target_train, target_test = \
train_test_split(data, target, test_size=0.33, random_state=42)

train_data = []
train_target = []
test_data = []
test_target = []
data =  StandardScaler().fit_transform(data)
#PCA dimensionality reduction
dimRed = decomposition.PCA(n_components=8)
dimRed.fit(data)
data =dimRed.transform(data)

# K fold cross validation
kf = KFold(n_splits=2,shuffle=True, random_state=1)
kf.get_n_splits(data, target)
for train_index, test_index in kf.split(data):
    train_data.append(data[train_index])
    train_target.append(target[train_index])
    test_data.append(data[test_index])
    test_target.append(target[test_index])
 
#names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
 #        "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
  #       "Naive Bayes", "QDA"]

names = [ "Decision Tree", "RandomForestClassifier",
         "Naive Bayes", "Neural Net"]
    
classifiers = [
    #AdaBoostClassifier(base_estimator=KNeighborsClassifier(3),n_estimators=50,
     #                  learning_rate=1.0, algorithm='SAMME.R', random_state=None),
                       
    #AdaBoostClassifier(base_estimator=LinearSVC(), n_estimators=50, learning_rate=1.0,
     #                  algorithm='SAMME', random_state=None),
                       
    #AdaBoostClassifier(base_estimator=SVC(gamma=2, C=1, cache_size=7000),n_estimators=50,
                       #learning_rate=1.0, algorithm='SAMME', random_state=None),                  
    DecisionTreeClassifier(max_depth=20),
                       
    RandomForestClassifier(max_depth=20, n_estimators=10, max_features='auto'),
    GaussianNB(),
    MLPClassifier(max_iter=2000)
    #AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=20)),
    ]


for idx, clf in enumerate(classifiers):
    print("\n#########################################\n")
    print("CLASSIFIER:" + names[idx])
    print("\n#########################################\n")
    for i in range(len(train_data)):
        clf.fit(train_data[i], train_target[i].ravel())
        clf.predict(train_data[i])
        training_accuracy = clf.score(train_data[i],train_target[i].ravel())
        test_accuracy = clf.score(test_data[i], test_target[i].ravel())
        print("Training accuracy for cross validation set  "+str(i+1)+": "+
               str(training_accuracy))
        print("Testing accuracy for cross validation set  "+str(i+1)+": "+
               str(test_accuracy))

