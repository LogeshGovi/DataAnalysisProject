import numpy as np
from .RandomSampling import RandomSampling as rs

class ClusterSampling:

    def cluster_sample(np_array, label_arr, per_sample, replacement=True):
        np.random.seed(15)
        unique_labels = np.unique(label_arr)
        j = 0
        for i in unique_labels:
            label_idx = np.where(label_arr[:]==i)[0]
            cluster_arr = np_array[label_idx,:]
            target_arr = label_arr[label_idx]
            sample_arr, target = rs.get_random_samples_np(cluster_arr,target_arr,per_sample,replace=replacement)
            if j == 0:
                sample = sample_arr
                sample_target = target
            else:
                sample = np.append(sample,sample_arr,axis=0)
                sample_target = np.append(sample_target,target,axis=0)
            j=j+1
        return sample, sample_target


