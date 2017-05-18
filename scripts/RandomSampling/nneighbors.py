
from anapy.externals.LoadDataset import LoadDataset as ld
from anapy.mlops.learnfromsample import LearnFromSample as lfs
import pandas as pd
import random
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

random.seed(15)
file_path = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\"
train_file = file_path+"training_set1.dat"
test_file = file_path+"testing_set1.dat"
scaler_file = file_path+"standardscaler.dat"

training_dataset = ld.load_pickle(train_file)
testing_dataset = ld.load_pickle(test_file)
train_data, train_target = ld.data_target_separator(training_dataset)
test_data, test_target = ld.data_target_separator(testing_dataset)

with open(scaler_file,mode='rb') as h:
    data_scaler = joblib.load(h)

folder_to_write = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\kneighbors\\"
sample_sizes = [30,40,50,60,70,80,90,100]
sampling_methods = ['random', 'systematic','stratified', 'cluster']
kneighbor = KNeighborsClassifier(n_neighbors=5)

for k,i in enumerate(sampling_methods):
    train_acc_values = []
    test_acc_values = []
    time_fit_values = []
    time_predict_values = []
    for j in sample_sizes:
        train, test, time = lfs.learn_from_sample(train_data, train_target, 
                                            test_data, test_target, 30,
                                            i, kneighbor,data_scaler,False,5)
        train_acc = accuracy_score(train[0], train[1])
        test_acc = accuracy_score(test[0], test[1])
        write_to_file = False
        if write_to_file == True:
            train_report = precision_recall_fscore_support(train[0],train[1])
            train_report_df = pd.DataFrame(np.array(train_report).T, 
                                           index=[1,2,3,4,5,6,7,8,9],
                                           columns=['precision','recall','fscore','support'])
            train_report_df.to_html(folder_to_write+i+"_train_report_"+str(j)+".html")
            test_report = precision_recall_fscore_support(test[0], test[1])
            test_report_df = pd.DataFrame(np.array(test_report).T,
                                          index=[1,2,3,4,5,6,7,8,9],
                                          columns=['precision','recall','fscore','support'])
            test_report_df.to_html(folder_to_write+i+"_test_report_"+str(j)+".html")
        train_acc_values.append(train_acc)
        test_acc_values.append(test_acc)
        time_fit_values.append(time[0])
        time_predict_values.append(time[1])
        
        
    acc_time_values = np.concatenate((train_acc_values,
                                      test_acc_values,time_fit_values,
                                      time_predict_values),axis=0).reshape(4,8).T
    acc_time_values_df = pd.DataFrame(acc_time_values,index=sample_sizes,columns=['tracc','teacc','fittm','predtm'])
    acc_time_values_df.to_html(folder_to_write+"acc_time_"+i+".html")

"""
    plt.figure(k+1, figsize=(8,2))
    plt.plot(sample_sizes,train_acc_values,'go',sample_sizes,train_acc_values, 'g' )
    plt.plot(sample_sizes,test_acc_values,'ro',sample_sizes,test_acc_values, 'r')
    plt.ylim((np.min(test_acc_values)-0.02, np.max(train_acc_values)+0.02))
    plt.title("Training/Test Accuracy graph for sampling method: "+ i)
    plt.show()
    
    plt.figure(5)
    plt.plot(sample_sizes,time_fit_values,'o', sample_sizes,time_fit_values)
    plt.plot(sample_sizes,time_predict_values,'o', sample_sizes,time_predict_values)
    plt.title("Fit/Predict graph: ")
    plt.show()
    
    del(time_fit_values)
    del(time_predict_values)
    del(train_acc_values)
    del(test_acc_values)



    

"""




