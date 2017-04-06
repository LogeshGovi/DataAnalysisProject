# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:09:01 2017

@author: Logesh Govindarajulu
"""

from datamanip import RandomSampling
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random


np.random.seed(15)
a = np.random.randint(1,10,5)
df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\prelimMLdataset.csv",
                  dtype='unicode')

cols = [0,1,-1]
rows =[0]
df.drop(df.columns[cols],axis=1,inplace=True)
df_records = np.shape(df)[0]

#mean of the parent dataset for the values:
# Noise_level, latitude, longitude, temperature, level, weekday, hour, minute
df_columns = df.values.astype(float)
column_means = np.mean(df_columns, axis=0)
column_means = column_means[1:-1]
#removing the key column and the label column
variable_columns = np.delete(df_columns,0,axis=1)
variable_columns = np.delete(variable_columns,8,axis=1)
# Normal plot of the distribution of each column
normPlotTitles = ['NoiseLevel', 'Latitude', 'Longitude',
                  'Temperature', 'Level', 'WeekDay', 'Hour', 
                  'Minute']
for i in range(np.shape(variable_columns)[1]):
    h = sorted(variable_columns[:,i])
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))
    plt.figure(i+1)
    plt.plot(h,fit, color='b', linewidth=2)
    plt.grid(True)
    plt.title(normPlotTitles[i]+": mu: "+ str(np.mean(h))+" sigma: "+ str(np.std(h)))
    #plt.hist(h,normed=True,color='b')
    plt.show()
    
    
    
#Sampling without replacement
samplingPercent = 10
rs = RandomSampling.RandomSampling()
sample10 = rs.without_replacement(df_records,samplingPercent)

for j in sample10:
    sdf = pd.DataFrame(j, columns=["PK"],dtype='str')
    sample_df = pd.merge(sdf,df,how="left", on="PK")
    
