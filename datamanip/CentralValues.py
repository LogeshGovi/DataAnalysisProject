# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:34:46 2017

@author: Logesh Govindarajulu
"""

import numpy as np
#from scipy.stats import mstats
class CentralValues:
    _mean = None
    _median = None 
    _range = None
    _var = None
    _stdDev = None
    _percentile = None
    _quantile = None
        
        
    def __init__(self, distArr):
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
            
        #def distPercentile(self):
            #self.__percentile = np.percentile(self.distArr)    
        def distQuantileDist(self):
            var_quantile = np.percentile(self.distArr,np.arange(100))
            #quantiles = mstats.mquantiles(self.distArr)
            self._quantile = var_quantile
            
        
        distMean(self)
        distMedian(self) 
        distRange(self)
        distVar(self) 
        distStdDev(self)
        #distPercentile(self)
        distQuantileDist(self)
               
        cVal = {
                 "mean":self._mean,
                 "median":self._median,
                 "range":self._range,
                 "var" : self._var,
                 "stddev": self._stdDev,
                 "quantile" :self._quantile
                 }
        return cVal
             
        
        
                 
            
        

        
        