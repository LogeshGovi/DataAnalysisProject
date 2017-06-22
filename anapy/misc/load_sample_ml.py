import anapy.misc.ml_utility as mlu
from sklearn.base import ClassifierMixin
class LoadData:
    def __init__(self, trainfile, testfile, scalerfile):
        self.train = trainfile
        self.test = testfile
        self.scale = scalerfile

class SamplingMethod:
    def __init__(self, samp_size_arr, samp_method_arr):
        self.samp_size = samp_size_arr
        self.samp_method = samp_method_arr


class Load_N_Sample:
    def __init__(self, load_data_obj, samp_method_obj):
        if isinstance(load_data_obj,LoadData):
            self.load_data_obj = load_data_obj
        if isinstance(samp_method_obj,SamplingMethod):
            self.samp_method_obj = samp_method_obj

    def load_N_sample(self):
        train_dataset = mlu.load_dataset(self.load_data_obj.train)
        test_dataset = mlu.load_dataset(self.load_data_obj.test)
        data_scaler = mlu.load_scaler(self.load_data_obj.scale)
        ss, sm = mlu.samp_parameters(self.samp_method_obj.samp_size,self.samp_method_obj.samp_method)
        return train_dataset, test_dataset, data_scaler,ss,sm


class RunML:
    def __init__(self,load_n_sample,clf, folder_to_write):
        if isinstance(load_n_sample,Load_N_Sample):
            self.load_n_sample = load_n_sample
            self.train, self.test, self.scaler, self.ss, self.sm = load_n_sample.load_N_sample()
        if isinstance(clf,ClassifierMixin):
            self.clf = clf
            self.folder_to_write = folder_to_write

    def runML(self,write_file=True):
        mlu.train_eval(self.clf,self.sm,self.train,self.test,self.ss, self.scaler, self.folder_to_write,write_to_file=write_file)



if __name__ == "__main__":
    ld = LoadData()
