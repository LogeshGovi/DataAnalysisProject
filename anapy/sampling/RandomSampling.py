# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:37:06 2017

@author: Logesh Govindarajulu
"""

import numpy as np

class RandomSampling:

##########################################################################################################
# Random Sampling for pandas dataframes
##########################################################################################################
    def without_replacement_df(df_size, per_sample ):
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


    def with_replacement_df(df_size,per_sample,no_of_samples=10):
        np.random.seed(15)
        df_key = np.array(np.arange(1,df_size))
        no_of_observations = (df_size*per_sample)//100
        samples=[]
        for i in range(no_of_samples):
            sample= np.random.choice(df_key, size = no_of_observations,
                                     replace=False)
            samples.append(sample)
        return samples

##########################################################################################################



##########################################################################################################
# Random Sampling for numpy datasets
##########################################################################################################

    def get_random_samples_np(np_array, per_sample, replace = True):
        """
        This method takes a two dimensional numpy array and returns a random sample based
        on the percentage of the samples needed.
        :param per_sample: 0 - 100
        :param replace: True or False
        :return: numpy sample array
        """
        replacement = replace
        def with_replacement_np(np_array, per_sample):
            np.random.seed(15)
            # number of samples that are needed to be drawn
            no_of_observations = len(np_array)*per_sample//100
            # the row indices that are drawn as sample
            sample_rows = np.random.choice(np.arange(len(np_array)),no_of_observations,replace=True)
            sample = np_array[sample_rows,:]
            return sample


        def without_replacement_np(np_array, per_sample):
            np.random.seed(15)
            np.random.shuffle(np_array)
            #number of samples that are needed to be drawn
            no_of_observations = len(np_array)*per_sample//100
            sample = np_array[:no_of_observations,:]
            return sample

        if replacement == True:
           return list(with_replacement_np(np_array,per_sample))
        elif replacement == False:
           return list(without_replacement_np(np_array,per_sample))
##########################################################################################################
