# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:04:46 2017

@author: Logesh Govindarajulu
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#cross-validation
from sklearn.model_selection import KFold

#Base Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import  QuadraticDiscriminantAnalysis

#Dimensionality Reduction
from sklearn import decomposition
