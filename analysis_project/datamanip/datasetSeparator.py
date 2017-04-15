# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 22:49:14 2017

@author: Logesh Govindarajulu
"""

import pandas as pd

"""
This module has the class DataSeparator
The instance of this class can be constructed with a pandas dataframe as argument
"""

class DataSeparator:
   
    def __init__(self, df):
        self.df = df
        
  
    def displayCols(self):
        """
        displayCols function returns a dictionary with column headers and
        their positions
        """
        df = self.df
        colHeader = df.columns.values.tolist()
        colDict = {}
        for i, j in enumerate(colHeader):
            colDict[j] = i
            
        return colDict
            
        
    def remCols(self,colarr):
        """
        remCols function returns the dataframe with the removed columns
        The columns to be removed must be passed as a 
        (1)string list containing the column names to be removed or
        (2)integer list containing the index of the column names to be removed
        (3)default argument colarr is an empty list
        """
        df = self.df
        if all(type(n) is int for n in colarr):
            df.drop(df.columns[colarr],inplace=True,axis=1)
        elif all(type(n) is str for n in colarr):
            df.drop(colarr,inplace=True,axis=1)
        else:
            print("Column headers must be a string list or int list")
        return df
            
    
        
        
        

    def sep_data_target(self, df):
        """
        (1) This function assumes that the last column is the target column
        and the other columns contain the data
        
        (2)If there are columns that are not needed, then they can be removed
        using the remCols function of this class
        
        (3)The dataset is divided into data and target and converted into 
        numpy arrays
        
        returns a two numpy arrays, data and target
        
        The argument has_header takes boolean values True or False
        """
        rdf = pd.DataFrame([])
        rdf = df.copy(deep=True)
        
        datadf = pd.DataFrame([])
        rdf.drop(rdf.columns[[-1]],axis=1,inplace=True)
        datadf = rdf.copy(deep=True)
        
        targetdf = pd.DataFrame([])
        targetdf = rdf[[-1]]
        datadf = datadf.as_matrix()
        targetdf= targetdf.as_matrix()
        
        return datadf, targetdf
        
        

    
