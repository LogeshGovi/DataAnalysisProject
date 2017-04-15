# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:52:35 2017

@author: Logesh Govindarajulu
"""

import unittest 
import pandas as pd
import numpy as np
from analysis_project.analysis_project.datamanip.datasetSeparator import DataSeparator

class TestdatasetSeparator(unittest.TestCase):
    testdf = pd.DataFrame(
                           np.array([[1,2],[1,3],[1,4],[1,5]]),
                           columns=['a','b']
                         )  
    ds1 = DataSeparator(testdf)
    result = ds1.displayCols()
    
    def testDisplayCols(self):
        ds = DataSeparator(self.testdf)
        self.assertEqual(ds.displayCols(),self.result)
            


if __name__ == '__main__':
    unittest.main()
