# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:59:32 2017

@author: Simone
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Panel, Series
from string import ascii_lowercase as letters
from scipy.stats import chisqprob
xs = Series(np.arange(10),index=tuple(letters[:10]))

