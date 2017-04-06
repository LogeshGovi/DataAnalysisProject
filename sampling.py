# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:37:06 2017

@author: Logesh Govindarajulu
"""

import numpy as np
from random import randint
import collections
a = set()
b = set()
c=[]
d=[]
k = 0
x = 170800

while len(a)<np.int(0.5*x):
    y=randint(1,170800)
    a.add(y)
    
i=0
while len(b)<np.int(0.5*x):
    y=randint(1,170800)
    if y not in a:
        b.add(y)
   