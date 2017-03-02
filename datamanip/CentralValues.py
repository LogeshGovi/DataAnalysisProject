# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:34:46 2017

@author: Logesh Govindarajulu
"""

import numpy as np

class CentralValues(): 
"""
"""       
    def __init__(self, distArr):
        if distArr.dtype == "float" or "int":
            if len(np.shape(distArr)) == 1:
                self.distArr = distArr
            else:
                print("distribution array must have a tuple of length 1")
            
        else:
            print("given numpy array is not an integer array")
        
    def distMean(self):
        return np.mean(self.distArr)
        
    def distMedian(self):
        return np.median(self.distArr)
        
    def distRange(self):
        return (np.max(self.distArr)-np.min(self.distArr))
    
    def distVar(self):
        return np.var(self.distArr)
        
    def distStdDev(self):
        return np.std(self.distArr)
        
    def distPercentile(self):
        return np.percentile(self.distArr)
        
        
        