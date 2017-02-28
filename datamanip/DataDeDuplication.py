# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:44:04 2017

@author: Logesh Govindarajulu
"""
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
