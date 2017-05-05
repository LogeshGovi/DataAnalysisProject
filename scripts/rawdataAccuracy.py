
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold



from anapy.datamanip import datasetSeparator as ds

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

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
data_lc_std = StandardScaler().fit_transform(data_lc)
target_lc = target_lc.astype('float')
target_lc = np.ravel(target_lc)
#####################################################################################################


#######################################################################################################
# Process of splitting the data into training set and test set
#######################################################################################################
# K fold cross validation
# create an instance of the KFold class
# number of splits = 5  so that the data is split as 80% and 20%
# the data is shuffled
train_data, train_target, test_data, test_target = [], [], [], []

kf = KFold(n_splits=5, shuffle=True, random_state=1)

dataset_number = 0
file_path = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\"
for train_index, test_index in kf.split(data_lc_std):
    train_data.append(data_lc_std[train_index])
    train_target.append(target_lc[train_index])
    test_data.append(data_lc_std[test_index])
    test_target.append(target_lc[test_index])
    # the following code writes the training set and testing set to files
    with open(file_path+'training_set'+str(dataset_number+1)+".dat",'wb') as f:
        pickle.dump(zip(train_data,train_target), f)

    with open(file_path+'testing_set'+str(dataset_number+1)+".dat",'wb') as g:
        pickle.dump(zip(test_data,test_target),g)

    dataset_number = dataset_number + 1
#####################################################################################################


#######################################################################################################
# Machine learning Algorithms --- Simple Classifiers
#######################################################################################################
ml_algorithms = [
                  "Nearest Neighbors",
                  "Decision Tree",
                  "Gaussian NB",
                  "Neural Net"
                ]
trainacc =[]
testacc = []
for num,ele in enumerate(train_data):
    for neighbors in range(12):
            knn = KNeighborsClassifier(n_neighbors= neighbors+1 ,weights='uniform',metric='minkowski')
            knn.fit(train_data[num],train_target[num])
            train_accuracy = knn.score(train_data[num],train_target[num])
            test_accuracy = knn.score(test_data[num],test_target[num])
            print("training accurracy -- traindata no. "+ str(num) +"kneighbor: "+ str(neighbors) + str(train_accuracy))
            trainacc.append(train_accuracy)
            testacc.append(test_accuracy)
    traintestacc = zip(trainacc,testacc)
    traintestacc_df = pd.DataFrame(traintestacc)
    traintestacc_df.to_html(file_path+'crossvalidationset_'+str(num)+".html")
    print("Training and Testing accuracy for  "+ str(num)+ "  of K Neighbors")
    plt.figure(num=num)
    plt.plot(np.arange(1,np.shape(trainacc)[0]+1), trainacc)
    plt.plot(np.arange(1,np.shape(testacc)[0]+1), testacc)
    plt.show()
    del trainacc[:]
    del testacc[:]



dtc =DecisionTreeClassifier(max_depth=12)
naive_bayes=GaussianNB()
neural_net = MLPClassifier(max_iter=500)
