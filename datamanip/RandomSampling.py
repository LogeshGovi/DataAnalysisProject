# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:37:06 2017

@author: Logesh Govindarajulu
"""

import numpy as np

class RandomSampling:
    
    def without_replacement(self,df_size, per_sample ):
        np.random.seed(15)
        df_key = np.array(np.arange(1,df_size))
        no_of_samples = 100//per_sample
        no_of_observations = (df_size*per_sample)//100
        samples = []
        for i in range(no_of_samples):
            sample = np.random.choice(df_key,size=no_of_observations,
                                       replace=False)
            df_key = np.setdiff1d(df_key,sample)
            samples.append(sample)
        return samples
                                   

    def with_replacement(self,df_size,per_sample,no_of_samples=10):
        np.random.seed(15)
        df_key = np.array(np.arange(1,df_size))
        no_of_observations = (df_size*per_sample)//100
        samples=[]
        for i in range(no_of_samples):
            sample= np.random.choice(df_key, size = no_of_observations, 
                                     replace=False)
            samples.append(sample) 
        return samples
                                     
                                     
