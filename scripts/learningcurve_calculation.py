from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.learning_curve import validation_curve

import matplotlib.pyplot as plt

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
out_columns = [np.shape(train_lc)[1]-1]
in_columns = [i for i in range(np.shape(train_lc)[1]) if i not in out_columns]
train_data, train_target = train_lc[:,in_columns], train_lc[:,-1]
#train_target = np.reshape(train_target,(np.shape(train_target)[0],1))





