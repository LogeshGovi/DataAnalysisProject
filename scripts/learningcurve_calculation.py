from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import numpy as np

random.seed(1)
train_file = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\training_set1.dat"
test_file = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\testing_set1.dat"
with open(train_file,mode='rb') as f:
    train_lc = pickle.load(f)

with open(test_file,mode='rb') as g:
    test_lc = pickle.load(g)


np.random.shuffle(train_lc)
out_columns_train = [np.shape(train_lc)[1]-1]
in_columns_train = [i for i in range(np.shape(train_lc)[1]) if i not in out_columns_train]
train_data, train_target = train_lc[:,in_columns_train], train_lc[:,-1]

np.random.shuffle(test_lc)
out_columns_test = [np.shape(test_lc)[1]-1]
in_columns_test = [i for i in range(np.shape(test_lc)[1]) if i not in out_columns_test]
test_data, test_target = test_lc[:,in_columns_test], test_lc[:,-1]
#train_target = np.reshape(train_target,(np.shape(train_target)[0],1))

DTC = DecisionTreeClassifier(max_depth=31)
DTC.fit(train_data,train_target)
y_pred = DTC.predict(test_data)
y_true = test_target

#metrics
class_report = np.array(precision_recall_fscore_support(y_true,y_pred)).T
class_report_df = pd.DataFrame(class_report, index = [1,2,3,4,5,6,7,8,9],
                               columns=['precision','recall','fscore','support'])
