import numpy as np
from .RandomSampling import RandomSampling as rs

class StratifiedSampling:

    def stratified_sample(np_array, strat_col, per_sample, replacement=True):
        np.random.seed(15)
        strat_set = np.unique(np_array[:,strat_col])
        j = 0
        for i in strat_set:
            strat_idx = np.where(np_array[:,strat_col]==i)[0]
            strat_arr = np_array[strat_idx,:]
            sample_arr = rs.get_random_samples_np(strat_arr,per_sample,replace=replacement)
            if j == 0:
                sample = sample_arr
            else:
                sample = np.append(sample,sample_arr,axis=0)
            j=j+1
        return sample


