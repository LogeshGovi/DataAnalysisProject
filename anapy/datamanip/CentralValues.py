# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:34:46 2017

@author: Logesh Govindarajulu
"""

import numpy as np
"""
Class to calculate the central values of a data set distribution
Values included are arithmetic mean, median, range, variance, standard deviation, and quantile
"""
class CentralValues:
    _mean = None
    _median = None 
    _range = None
    _var = None
    _stdDev = None
    _quantile = None


    def __init__(self, distArr):
        # data set is passed as a numpy array
        if distArr.dtype == "float" or "int":
            if len(np.shape(distArr)) == 1:
                self.distArr = distArr
            else:
                print("distribution array must have a tuple of length 1")
            
        else:
            print("given numpy array is not an integer array")
    
    def cenvalues(self):    
        def distMean(self):
            self._mean = np.mean(self.distArr)
            
        
        def distMedian(self):
            self._median = np.median(self.distArr)
            
        def distRange(self):
            self._range = (np.max(self.distArr)-np.min(self.distArr))
        
        def distVar(self):
            self._var = np.var(self.distArr)
            
        def distStdDev(self):
            self._stdDev = np.std(self.distArr)
            

        def distQuantileDist(self):
            var_quantile = np.percentile(self.distArr,np.arange(100))
            self._quantile = var_quantile
            
        
        distMean(self)
        distMedian(self) 
        distRange(self)
        distVar(self) 
        distStdDev(self)
        distQuantileDist(self)

        # returns a dictionary as given the cVal variable
        cVal = {
                 "mean":self._mean,
                 "median":self._median,
                 "range":self._range,
                 "var" : self._var,
                 "stddev": self._stdDev,
                 "quantile" :self._quantile
                 }
        return cVal
             
        
        
                 
            
        

        
