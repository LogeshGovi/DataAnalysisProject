
from anapy.misc import ml_utility as mlu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

train_dataset = mlu.load_dataset("D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\training_set1.dat")
test_dataset = mlu.load_dataset("D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\testing_set1.dat")
data_scaler = mlu.load_scaler("D:\\lcif\\16032017-IndividualFiles\\Chess-King-Rook-King\\standardscaler.dat")

ss, sm = mlu.samp_parameters([30,40,50,60,70,80,90,100],['random', 'systematic','stratified', 'cluster'])
#dtc = DecisionTreeClassifier(criterion='gini',max_depth=21)
#gnb = GaussianNB()
#neuralnet = MLPClassifier(max_iter=500)
kneighbors = KNeighborsClassifier(n_neighbors=5)
clf = AdaBoostClassifier(base_estimator=kneighbors,n_estimators=50)
folder_write = "D:\\lcif\\16032017-IndividualFiles\\TrainTestdataset\\adaboost\\kneighbors\\"
mlu.train_eval(clf,sm,train_dataset, test_dataset,ss,data_scaler, folder_write, write_to_file=True)
