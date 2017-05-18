import pickle
import numpy as np

class LoadDataset:
    def load_pickle(file):
        """

        :return: dataset a numpy array
        """
        with open(file,mode='rb') as f:
            dataset = pickle.load(f)
        return dataset

    def data_target_separator(np_dataset):
        np.random.shuffle(np_dataset)
        out_columns = [np.shape(np_dataset)[1]-1]
        in_columns = [i for i in range(np.shape(np_dataset)[1]) if i not in out_columns]
        data, target = np_dataset[:,in_columns], np_dataset[:,-1]
        return data, target

