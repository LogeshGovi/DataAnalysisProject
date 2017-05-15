from fractions import Fraction
import numpy as np
import pandas as pd

class SystematicSampling:

    def systematic_sample(np_array, target_array, per_sample):
        np.random.seed(15)
        #number of samples that are to be drawn
        sample_block = Fraction(per_sample,100)
        no_samples_per_block = sample_block.numerator
        sample_block_size = sample_block.denominator
        sample_selection_start = np.random.choice(np.arange(sample_block_size+1))
        initial_sample_drawn = np.arange(sample_selection_start,sample_selection_start+no_samples_per_block,step=1)
        row_idx = np.array([])
        for i in initial_sample_drawn:
            samples_idx = np.arange(i,len(np_array),step=sample_block_size)
            row_idx = np.concatenate((row_idx,samples_idx),axis=0)
        row_idx = row_idx.astype(int)
        sample = np_array[row_idx,:]
        sample_target = target_array[row_idx]
        return sample, sample_target

