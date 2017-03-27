# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:45:18 2017

@author: Logesh Govindarajulu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#import matplotlib.backends.backend_pdf as bpdf
plt.ioff()


df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\FJoinTSDiscreteValuesFilled.csv",
                  dtype='unicode')
                  

              
app_list = sorted(df['app']
           .drop_duplicates().dropna()
           .values.astype('U'))   
           
appDataSet = []
           
for app in app_list:
    adf = df.loc[df['app']==app]
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
   
   
   app_df = noise.loc[:,['app']].drop_duplicates()
   appName = app_df.values.astype(str)[0][0]
   appList.append(appName)   
   
   lathist = lat['lat'].dropna().values.astype('double')
   lnghist = lng['lng'].dropna().values.astype('double')
   temphist = temp['temp'].dropna().values.astype('double')
   levelhist = level['level'].dropna().values.astype('double')
   noisehist = noise['noise_level_db'].dropna().values.astype('double')
   
   # Plotting the histogram of all the noise level data values for each app
   plt.figure(j)
   plt.hist(noisehist,bins=10)
   plt.title(appName)
   plt.xlabel('noise_level_db')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\full\\NoiseLevel\\"+appName+".png")
   plt.close()
   ########################################################################
   
   # Plotting the histogram of all the latitude values for each app
   plt.figure(j)
   plt.hist(lathist,bins=10)
   plt.title(appName)
   plt.xlabel('Latitude')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\full\\Latitude\\"+appName+".png")
   plt.close()
   ########################################################################   
   
    # Plotting the histogram of all the longitude values for each app
   plt.figure(j)
   plt.hist(lnghist,bins=10)
   plt.title(appName)
   plt.xlabel('Longitude')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\full\\Longitude\\"+appName+".png")
   plt.close()
   ########################################################################   
  
   # Plotting the histogram of all the level values for each app
   plt.figure(j)
   plt.hist(levelhist,bins=10)
   plt.title(appName)
   plt.xlabel('Level')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\full\\Level\\"+appName+".png")
   plt.close()
   ########################################################################   
   
    # Plotting the histogram of all the latitude values for each app
   plt.figure(j)
   plt.hist(temphist,bins=10)
   plt.title(appName)
   plt.xlabel('Temperature')
   plt.ylabel('Frequency')
   plt.savefig("D:\\lcif\\Histogram\\full\\Temperature\\"+appName+".png")
   plt.close()
   ########################################################################   
  
  
   

   j=j+1