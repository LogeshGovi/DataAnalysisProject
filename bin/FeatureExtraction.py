# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:04:28 2017
@author: Logesh Govindarajulu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from scipy.stats import norm, mode, mstats, linregress
from scipy.optimize import curve_fit

dataset = pd.read_csv("D:\\DedupCleanData.csv", 
            sep=',', dtype='unicode')
           
#print("The number of  columns in the dataframe is ", dataset.shape[1])
#print("The number of rows in the dataframe is ", dataset.shape[0])            
column = dataset[['Entity_Key', 'Entity_Value']]      
entity_freq = column.groupby('Entity_Key').count()
# save the entity frequencies in an html table
entity_freq.to_html('D:\\lcif\\RawDatasetFeatures\\FrequencyTable.html')
yVal = entity_freq[['Entity_Value']].values
xHeaders = entity_freq.index.values
xhead = xHeaders.reshape(np.size(xHeaders),1)
xVal = np.arange(np.size(yVal))
slope, intercept, r_value, p_value, std_err = linregress(xVal, yVal.reshape(np.size(yVal),))
colors = np.random.rand(len(yVal))
s = [10000*n/np.max(yVal) for n in yVal]
plt.figure(figsize = (10,10))
plt.scatter(xVal,yVal, c=colors,s=s,alpha=0.5)
plt.plot(xVal,(slope*xVal+intercept), color="blue")
for i, txt in enumerate(xHeaders):
    plt.annotate(txt,(xVal[i],yVal[i]),rotation=90)
plt.savefig("D:\\lcif\\RawDatasetFeatures\\RegressionMap.png")
plt.show()
xVal = xVal + 1

records = np.concatenate((xhead,yVal),axis=1)
records_sorted = records[records[:,1].argsort()[::1]]
print(records_sorted)

#Calculate the central values of the frequency distribution of the variables
var_max, var_min, var_range, var_mean, var_median, var_mode, var_variance, var_stddev = \
np.max(yVal), np.min(yVal), (np.max(yVal)-np.min(yVal)), np.mean(yVal), np.median(yVal),mode(yVal), np.var(yVal),np.std(yVal)
print("Max: %s\nMin:%s\nRange:%s\nMean: %s\nMedian: %s\nMode: %s\nVariance: %s\nstd dev: %s" 
      %(str(var_max), str(var_min), str(var_range),str(var_mean), str(var_median), str(var_mode[0][0]), str(var_variance), str(var_stddev)))

#remove the rows containing marginal variable frequencies
for colrm in records_sorted[records_sorted[:,1]>np.median(records_sorted[:,1])]:
    red_dataset = dataset[dataset.Entity_Key==colrm[0]]
    
# percentile calculation
var_quantile = np.percentile(yVal,np.arange(100))
quantiles = mstats.mquantiles(yVal)


#plot normal distribution 
plt.figure(1)
#plt.plot(norm.pdf(var_quantile,np.mean(var_quantile), np.std(var_quantile)))
'''for i, q in enumerate(quantiles):
    labels = ['25%', '50%', '75%']
    plt.plot(q, label = labels[i])
    '''
plt.boxplot(yVal)
plt.savefig('D:\\lcif\\RawDatasetFeatures\\boxplot.png')


"""
plt.figure(2, figsize=[20,8])
plt.bar(np.arange(np.size(var_quantile)),np.sort(var_quantile))
plt.savefig('percentilerange.png')
"""
def findVariableFreq(dataframe, ixColName,schColName, varName):
    varVal = dataframe.loc[dataframe[ixColName] == varName, schColName]
    varVal = varVal.dropna()
    freq =  varVal.value_counts(sort=True, ascending=False)
    return freq.values, freq.index.values
    
def plotFrequencies_Bar(fignum, x, y, xheader):
    fig = plt.figure(fignum, figsize=[10,8])
    ax = fig.add_subplot(111)
    ax.set_xlim(-1,50)
    ax.bar(x,y, align ='center')
    plt.xticks(x,xheader)
    plt.setp(ax.get_xticklabels(), fontsize=14, rotation='vertical')
    plt.show()
    plt.savefig('D:\\lcif\\RawDatasetFeatures\\EntityFreqHistogram.png')
    

def plotFrequencies_scatter(fignum, x, y, xheader):
    fig = plt.figure(fignum, figsize=[20,8])
    ax = fig.add_subplot(111)
    ax.set_xlim(-1,350)
    ax.scatter(x,y)
    plt.xticks(x,xheader)
    plt.setp(ax.get_xticklabels(), fontsize=14, rotation='vertical')
    plt.savefig('D:\\lcif\\RawDatasetFeatures\\EntityFreqScatter.png')
    
    
a, b  = findVariableFreq(column, "Entity_Key", "Entity_Value", "app")
#b = b.reshape(np.size(b),1)
a = a.reshape(np.size(a),1)
y = np.arange(np.size(b))


plotFrequencies_Bar(3,xVal,yVal,xHeaders)
#plotFrequencies_Bar(3,y,a,b)
