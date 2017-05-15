import pickle
import random
import numpy as np


from sklearn.externals import joblib
from anapy.sampling.RandomSampling import RandomSampling as rs
from anapy.sampling.SystematicSampling import SystematicSampling as ss
from anapy.sampling.ClusterSampling import ClusterSampling as cs

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






random_sample = rs.get_random_samples_np(train_data,50, True)
standardized_sample = data_scaler.transform(random_sample)
