# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:35:32 2017

@author: Logesh Govindarajulu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity
#import matplotlib.backends.backend_pdf as bpdf
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
    

appList = []    
noise_ds = []
noise_ds_nan = []
lat_ds = []
lng_ds = []
lat_ds_nan =[]
lng_ds_nan =[]
temp_ds = []
temp_ds_nan = []
level_ds = []
level_ds_nan = []

j=0
for ds in appDataSet:
    # select the columns PK, app and noise_level_db
   noise = (ds.loc[:,['PK','app','noise_level_db']])
   lat = (ds.loc[:,['PK','app','lat']])
   lng = (ds.loc[:,['PK','app','lng']])
   temp = (ds.loc[:,['PK','app','temp']])
   level = (ds.loc[:,['PK','app','level']])
   
   lathist = lat['lat'].dropna().values.astype('double')
   lnghist = lng['lng'].dropna().values.astype('double')
   temphist = temp['temp'].dropna().values.astype('double')
   levelhist = level['level'].dropna().values.astype('double')
   noisehist = noise['noise_level_db'].dropna().values.astype('double')
   
   app_df = noise.loc[:,['app']].drop_duplicates()
   appName = app_df.values.astype(str)[0][0]
   appList.append(appName)   
   
   # Plotting the histogram of all the noise level data values for each app
   plt.figure(j)
   plt.hist(noisehist, bins=10, normed=True)
   plt.title(appName)
   plt.xlabel('noise_level_db')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\NoiseLevel\\"+appName+".png")
   plt.close()
   ########################################################################
   
   # Plotting the histogram of all the latitude values for each app
   plt.figure(j)
   plt.hist(lathist, bins=10, normed=True)
   plt.title(appName)
   plt.xlabel('Latitude')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\Latitude\\"+appName+".png")
   plt.close()
   ########################################################################   
   
    # Plotting the histogram of all the longitude values for each app
   plt.figure(j)
   plt.hist(lnghist, bins=10, normed=True)
   plt.title(appName)
   plt.xlabel('Longitude')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\Longitude\\"+appName+".png")
   plt.close()
   ########################################################################   
  
   # Plotting the histogram of all the level values for each app
   plt.figure(j)
   plt.hist(levelhist, bins=10, normed=True)
   plt.title(appName)
   plt.xlabel('Level')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\Level\\"+appName+".png")
   plt.close()
   ########################################################################   
   
    # Plotting the histogram of all the latitude values for each app
   plt.figure(j)
   plt.hist(temphist, bins=10, normed=True)
   plt.title(appName)
   plt.xlabel('Temperature')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\Temperature\\"+appName+".png")
   plt.close()
   ########################################################################   
  
  
   
   
   
   #separate the non-null values of the variable data fields
   noise_ds.append(noise.dropna())
   lat_ds.append(lat.dropna())
   lng_ds.append(lng.dropna())
   temp_ds.append(temp.dropna())
   level_ds.append(level.dropna())
   
   #separate the nan values of the variable data fields
   noise_ds_nan.append(noise.loc[pd.isnull(noise['noise_level_db'])])
   lat_ds_nan.append(lat.loc[pd.isnull(lat['lat'])])
   lng_ds_nan.append(lng.loc[pd.isnull(lng['lng'])])
   temp_ds_nan.append(temp.loc[pd.isnull(temp['temp'])])
   level_ds_nan.append(level.loc[pd.isnull(level['level'])])
   j=j+1
   
 
appdf = pd.DataFrame(appList)
appdf.to_html('D:\\lcif\\RawDatasetFeatures\\totalapps.html')
i=0
"""
for i in range(len(noise_ds)):
    if np.shape(noise_ds[i])[0]==0 or np.shape(noise_ds_nan[i])[0]==0:
        i=i+1
    if np.shape(noise_ds[i])[0]>0 and np.shape(noise_ds_nan[i])[0]>0:
        noise_series = noise_ds[i]['noise_level_db'].values.astype('double')
        maxim = np.max(noise_series)
        minim = np.min(noise_series)
        noise_mode = stats.mode(noise_series)[0][0]
        ssize = np.shape(noise_ds_nan[i])[0]
        n_dist = np.random.uniform(minim,maxim,ssize)
        n_dist = n_dist
        noise_ds_nan[i]['noise_level_db']= \
        pd.Series(n_dist,index=noise_ds_nan[i].index)
    i=i+1
"""
##noise_nan = rdf.loc[pd.isnull(rdf['noise_level_db'])]
    

