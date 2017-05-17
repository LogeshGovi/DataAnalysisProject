from anapy.sampling.RandomSampling import RandomSampling as rs
import numpy as np
import random
import pickle
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

random.seed(15)
file_path = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\"
train_file = file_path+"training_set1.dat"
test_file = file_path+"testing_set1.dat"
scaler_file = file_path+"standardscaler.dat"

with open(train_file,mode='rb') as f:
    train_lc = pickle.load(f)

np.random.shuffle(train_lc)
out_columns_train = [np.shape(train_lc)[1]-1]
in_columns_train = [i for i in range(np.shape(train_lc)[1]) if i not in out_columns_train]
train_data, train_target = train_lc[:,in_columns_train], train_lc[:,-1]

with open(test_file,mode='rb') as g:
    test_lc = pickle.load(g)

np.random.shuffle(test_lc)
out_columns_test = [np.shape(test_lc)[1]-1]
in_columns_test = [i for i in range(np.shape(test_lc)[1]) if i not in out_columns_test]
test_data, test_target = test_lc[:,in_columns_test], test_lc[:,-1]

with open(scaler_file,mode='rb') as h:
    data_scaler = joblib.load(h)

sample_sizes = [30,40,50,60,70,80,90,100]


for i in sample_sizes:
    random_sample, sample_train_target = rs.get_random_samples_np(train_data, train_target,i, True)
    #out_columns_train = [np.shape(random_sample)[1]-1]
    #in_columns_train = [i for i in range(np.shape(random_sample)[1]) if i not in out_columns_train]
    #sample_train_data, sample_train_target = random_sample[:,in_columns_train], random_sample[:,-1]

    standardized_random_sample = data_scaler.transform(random_sample)
    standardized_random_sample = random_sample

    clf = KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='minkowski')
    clf.fit(standardized_random_sample,sample_train_target)
    training_acc = clf.score(standardized_random_sample,sample_train_target)
    test_acc = clf.score(test_data,test_target)
    print("Training accuracy| Sample size "+str(i)+"% : " + str(training_acc))
    print("Testing accuracy| Sample size "+str(i)+"% : " + str(test_acc))
    

clf = KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='minkowski')
#train_data = data_scaler.transform(train_data)
clf.fit(train_data,train_target)
training_acc = clf.score(train_data,train_target)
test_acc = clf.score(test_data,test_target)
print(training_acc)
print(test_acc)
