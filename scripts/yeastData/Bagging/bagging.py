
from anapy.externals.LoadDataset import LoadDataset as ld
from anapy.mlops.learnfromsample import LearnFromSample as lfs
import pandas as pd
import random
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# when write_to_file is set to True it writes classification report and time taken to fit and predict
# to the given file
write_to_file = True
random.seed(15)

#######################################################################################################################
# give the file path from where the train and test files can be loaded -- in pickle form
#######################################################################################################################
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

folder_to_write = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\bagging\\kneighbors\\"
sample_sizes = [30,40,50,60,70,80,90,100]
sampling_methods = ['random', 'systematic','stratified', 'cluster']
#sampling_methods =['systematic']
#dtc = DecisionTreeClassifier(max_depth=21)
#gnb = GaussianNB()
#neuralnet = MLPClassifier(max_iter=500)
kneighbors = KNeighborsClassifier(n_neighbors=5)
bagging = BaggingClassifier(base_estimator=kneighbors,n_estimators=10,
                            warm_start=False,max_samples=1.0)


total_train_acc = []
total_test_acc = []
total_time_fit = []
total_time_pred = []
for k,i in enumerate(sampling_methods):
    train_acc_values = []
    test_acc_values = []
    time_fit_values = []
    time_predict_values = []
    for j in sample_sizes:
        train, test, time = lfs.learn_from_sample(train_data, train_target,
                                            test_data, test_target, j,
                                            i, bagging,data_scaler,False,5)
        train_acc = accuracy_score(train[0], train[1])
        test_acc = accuracy_score(test[0], test[1])
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

    if write_to_file == True:
        acc_time_values = np.concatenate((train_acc_values,
                                          test_acc_values,time_fit_values,
                                          time_predict_values),axis=0).reshape(4,8).T
        acc_time_values_df = pd.DataFrame(acc_time_values,index=sample_sizes,columns=['tracc','teacc','fittm','predtm'])
        acc_time_values_df.to_html(folder_to_write+"acc_time_"+i+".html")

    total_train_acc.append(train_acc_values)
    total_test_acc.append(test_acc_values)
    total_time_fit.append(time_fit_values)
    total_time_pred.append(time_predict_values)

    plt.figure(k+1)
    plt.plot(sample_sizes,train_acc_values, 'g',
             sample_sizes,test_acc_values, 'r')
    plt.legend(('train accuracy','test accuracy'),
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim((0,1))
    plt.xlabel("Sample size in percentage")
    plt.ylabel("Accuracy")
    plt.title("Training/Test Accuracy graph for sampling method: "+ i)
    plt.show()

    """
    plt.figure(k+5)
    plt.plot(sample_sizes,time_fit_values,'go', sample_sizes,time_fit_values,'k')
    plt.plot(sample_sizes,time_predict_values,'ro', sample_sizes,time_predict_values,'k')
    plt.title("Fit/Predict graph: "+i)
    plt.show()
    """

    del(time_fit_values)
    del(time_predict_values)
    del(train_acc_values)
    del(test_acc_values)


plt.figure()
plt.plot(sample_sizes, total_train_acc[0],'r:',
         sample_sizes,total_train_acc[1],'g:',
         sample_sizes,total_train_acc[2],'b:',
         sample_sizes,total_train_acc[3],'m:',
         sample_sizes, total_test_acc[0],'r',
         sample_sizes,total_test_acc[1],'g',
         sample_sizes,total_test_acc[2],'b',
         sample_sizes,total_test_acc[3],'m',linewidth=5,alpha=0.7,)
plt.ylim(0,1)
plt.xlabel("Sample size in percentage")
plt.ylabel("Accuracy")
plt.title("Train/Test Accuracy for Different Sampling Methods")
plt.legend(("random/train","systematic/train","stratified/train","cluster/train",
            "random/test","systematic/test","stratified/test","cluster/test"),
            loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

plt.figure()
plt.plot(sample_sizes, total_time_fit[0],'r:',
         sample_sizes,total_time_fit[1],'g:',
         sample_sizes,total_time_fit[2],'b:',
         sample_sizes,total_time_fit[3],'m:',
         sample_sizes, total_time_pred[0],'r',
         sample_sizes,total_time_pred[1],'g',
         sample_sizes,total_time_pred[2],'b',
         sample_sizes,total_time_pred[3],'m')
plt.xlabel("Sample size in percentage")
plt.ylabel("Time in seconds")
plt.title("Fit/Predict time for Different Sampling Methods")
plt.legend(("random/fit","systematic/fit","stratified/fit","cluster/fit",
            "random/pred","systematic/pred","stratified/pred","cluster/pred"),
            loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()







