# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:49:01 2017

@author: Logesh Govindarajulu
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 

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

DTC = DecisionTreeClassifier()
DTC.fit(data_train, target_train)
DTC.predict(data_train)
training_accuracy = DTC.score(data_train,target_train)
test_accuracy = DTC.score(data_test, target_test)
print("Training accuracy is:  " + training_accuracy)
print("Testing accuracy is: "+ test_accuracy)

