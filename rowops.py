# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:35:32 2017

@author: Logesh Govindarajulu
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
plt.ioff()


rdf = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\FJoinonTimestamp.csv",
                  dtype='unicode')
              
app_list = sorted(rdf['app']
           .drop_duplicates().dropna()
           .values.astype('U'))   
           
appDataSet = []
           
for app in app_list:
    adf = rdf.loc[rdf['app']==app]
    appDataSet.append(adf)
    #adf.to_csv("D:\\lcif\\Appwisedatasets\\"+app+".csv", sep=",",
               #index=False, header=True)
noise_ds = []
noise_ds_nan = []
latlng_ds = []
latlng_ds_nan =[]
temp_ds = []
temp_ds_nan = []
level_ds = []
level_ds_nan = []

j=0
for ds in appDataSet:
   noise = (ds.loc[:,['PK','app','noise_level_db']])
   latlng = (ds.loc[:,['PK','app','lat','lng']])
   temp = (ds.loc[:,['PK','app','temp']])
   level = (ds.loc[:,['PK','app','level']])
   noisehist = noise['noise_level_db'].dropna().values.astype('double')
   appName = noise.loc[:,['app']].drop_duplicates().values.astype(str)[0][0]
   #if len(noisehist)>0:
   plt.figure(j)
   plt.hist(noisehist, bins=10)
   plt.title(appName)
   plt.xlabel('noise_level_db')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\"+appName+".png")
   noise_ds.append(noise.dropna())
   latlng_ds.append(latlng.dropna())
   temp_ds.append(temp.dropna())
   level_ds.append(level.dropna())
   
   noise_ds_nan.append(noise.loc[pd.isnull(noise['noise_level_db'])])
   latlng_ds_nan.append(latlng.loc[pd.isnull(latlng['lat'])])
   temp_ds_nan.append(temp.loc[pd.isnull(temp['temp'])])
   level_ds_nan.append(level.loc[pd.isnull(level['level'])])
   j=j+1
   
i=0
for i in range(len(noise_ds)):
    if np.shape(noise_ds[i])[0]==0 or np.shape(noise_ds_nan[i])[0]==0:
        i=i+1
    if np.shape(noise_ds[i])[0]>0 and np.shape(noise_ds_nan[i])[0]>0:
        maxim = np.max(noise_ds[i]['noise_level_db'].values.astype('double'))
        minim = np.min(noise_ds[i]['noise_level_db'].values.astype('double'))
        ssize = np.shape(noise_ds_nan[i])[0]
        n_dist = np.random.uniform(minim,maxim,ssize)
        n_dist = n_dist
        noise_ds_nan[i]['noise_level_db']= \
        pd.Series(n_dist,index=noise_ds_nan[i].index)
    i=i+1
       
##noise_nan = rdf.loc[pd.isnull(rdf['noise_level_db'])]
    

