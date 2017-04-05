# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:37:06 2017

@author: Logesh Govindarajulu
"""

import numpy as np
from random import randint
import collections
a = set()
c=[]
k = 0
x = 170800

while len(a)<np.int(x):
    b=randint(1,170800)
    a.add(b)
    
for j in a:
    c.append(j)
    
c.sort()

