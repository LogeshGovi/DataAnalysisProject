
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
import random


from anapy.datamanip import datasetSeparator as ds

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib

#######################################################################################################
# Preliminary steps for loading the data and splitting it into data and target
#######################################################################################################
# Load the raw machine learning data set
# learning context data -- mentioned as 'lc' with variables
raw_df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\prelimMLdatasetFull.csv",
                  dtype='unicode')

datalabel_lc = raw_df.ix[:,1].unique().astype(str)

#create an instance of data separator by passing the datafram raw_df as the argument
data_sep = ds.DataSeparator(raw_df)
df_lc = pd.DataFrame([])
# removal of the columns containing the text data of the labels and the final column with 25 labels
# as the 25 labels have been replaced with 10 labels for the purpose of simplicity
# only the columns from 4 to 12 consist of the actual machine learning set with labels being the
# last column
df_lc = data_sep.remCols([0,1,2,3,13])

# assuming the last column to be the target and the rest as data this method divides the data set
# into data and target labels
data_lc, target_lc = data_sep.sep_data_target(df_lc)
data_lc = data_lc.astype('float')
target_lc = target_lc.astype('float')
target_lc = np.ravel(target_lc)
full_data_set = np.insert(data_lc,np.shape(data_lc)[1],target_lc,axis=1)
np.random.shuffle(full_data_set)
#####################################################################################################



####################################################################################################
# Standard Scaler 
####################################################################################################
file_path = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\"
data_scaler = StandardScaler()
data_scaler.fit(data_lc)
with open(file_path+"standardscaler.dat",'wb') as h:
            joblib.dump(data_scaler,h)

####################################################################################################



#######################################################################################################
# Process of splitting the data into training set and test set
#######################################################################################################
# K fold cross validation
# create an instance of the KFold class
# number of splits = 5  so that the data is split as 80% and 20%
# the data is shuffled
train_data_array, train_target_array, test_data_array, test_target_array = [], [], [], []

kf = KFold(n_splits=5, shuffle=True)

dataset_number = 0
for train_index, test_index in kf.split(data_lc):
    train_data = data_lc[train_index].astype('float')
    train_target = target_lc[train_index].astype('float')
    train_target = np.reshape(train_target,(np.shape(train_target)[0],1))
    test_data = data_lc[test_index].astype('float')
    test_target = target_lc[test_index].astype('float')
    test_target = np.reshape(test_target,(np.shape(test_target)[0],1))

    train_data_array.append(train_data)
    train_target_array.append(train_target)
    test_data_array.append(test_data)
    test_target_array.append(test_target)

    # the following code writes the training set and testing set to files as an ndarray
    # if write_to_file value is set to True
    write_to_file = True
    if write_to_file == True:

        with open(file_path+'training_set'+str(dataset_number+1)+".dat",'wb') as f:
            train_data_target = np.concatenate((train_data,train_target),axis=1)
            pickle.dump(train_data_target, f)

        with open(file_path+'testing_set'+str(dataset_number+1)+".dat",'wb') as g:
            test_data_target = np.concatenate((test_data,test_target),axis=1)
            pickle.dump(test_data_target,g)

    dataset_number = dataset_number + 1
#####################################################################################################
