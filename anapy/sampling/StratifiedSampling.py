import numpy as np
from .RandomSampling import RandomSampling as rs

class StratifiedSampling:

    def stratified_sample(np_array, label_arr, strat_col, per_sample, replacement=True):
        np.random.seed(15)
        strat_set = np.unique(np_array[:,strat_col])
        j = 0
        for i in strat_set:
            strat_idx = np.where(np_array[:,strat_col]==i)[0]
            strat_arr = np_array[strat_idx,:]
            strat_target = label_arr[strat_idx]
            sample_arr, target = rs.get_random_samples_np(strat_arr,strat_target,per_sample,replace=replacement)
            if j == 0:
                sample = sample_arr
                sample_target = target
            else:
                sample = np.append(sample,sample_arr,axis=0)
                sample_target = np.append(sample_target,target,axis=0)
            j=j+1
        return sample, sample_target


