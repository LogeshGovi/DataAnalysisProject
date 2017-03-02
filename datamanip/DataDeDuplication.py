# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:44:04 2017

@author: Logesh Govindarajulu
"""
import numpy as np
import pandas as pd
import  os.path as fpath

class Deduplication:
    # This is a function to deduplicate the rows of a dataset
    def dataDedup_csv(infile, outfile=None):
        """ 
        This function takes a csv file as an argument
        deduplicates the file and writes the deduplicated 
        dataset to a csv file if a path for the output file is 
        provided as the second argument
        
        It returns the deduplicated dataframe
        
          ** Parameters** , **type**, **return values**
          :param infile: input csv file
          :param outfile: output csv file
          :type  arg1: input csv file path string
          :return: returns the deduplicated dataframe on success
        
        """
        if fpath.isfile(infile):
            
            dataset = pd.read_csv(infile, sep=',', dtype='unicode')
            dedup_dataset = dataset.drop_duplicates()
            
            if outfile!=None:
                dedup_dataset.to_csv(outfile, 
                         encoding='utf-8', index=False,
                         header=False)
                         
            return dedup_dataset
            
        else:
            print("file \"%s\" does not exist... or is not a file..." %(infile))




    # This is a function to split the dataframes such that each frame contains 
    # approximately 1 million records
    def dataFrameSplit(df, norec=1000000, outfile= None):
        """
    This function checks for the size of a dataframe and splits it into parts 
    containing approximately 1 million records as the default number of records
    for each dataframe.It also provides the option of writing the split 
    dataframes to the disk.
        
          ** Parameters** , **type**, **return values**
          :param df : dataframe to be split
          :param norec: number of records needed for each split *default 1000000*
          :type  arg1: pandas dataframe
          :type arg2: integer
          :return: returns an array of dataframes
    """
        # calculation of the no. of rows of the dataframe
        df_rsz = len(df.index)
        if df_rsz>norec:
            no_splits = np.ceil(df_rsz/norec)
            dfarr = np.array_split(df,no_splits)
            return dfarr
        else:
            print("The dataframe doesn't have sufficient records")
            
        # printing to disk when     
        if outfile!=None:
            i=0
            for arr in dfarr:
                arr.to_csv("D:\\ddf"+str(i+1)+".csv",encoding='utf-8', index=False,
                                     header=False)
                i = i+1
                    
        