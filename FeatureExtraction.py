# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:04:28 2017
@author: Simone
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from scipy.stats import norm, mode, mstats, linregress
from scipy.optimize import curve_fit


dataset = pd.read_csv("D:\\fulldataLearningContext.csv", 
            sep=',', dtype='unicode')
#print("The number of  columns in the dataframe is ", dataset.shape[1])
#print("The number of rows in the dataframe is ", dataset.shape[0])            
column = dataset[['Entity_Key', 'Entity_Value']]      
entity_freq = column.groupby('Entity_Key').count()
print(list(entity_freq.columns.values))
yVal = entity_freq[['Entity_Value']].values
xHeaders = entity_freq.index.values
xhead = xHeaders.reshape(46,1)
xVal = np.arange(np.size(yVal))
slope, intercept, r_value, p_value, std_err = linregress(xVal, yVal.reshape(46,))
plt.figure()
plt.scatter(xVal,yVal, color="k")
plt.plot(xVal,(slope*xVal+intercept), color="blue")
xVal = xVal + 1

records = np.concatenate((xhead,yVal),axis=1)
records_sorted = records[records[:,1].argsort()[::-1]]

#Calculate the central values of the frequency distribution of the variables
var_mean, var_median, var_mode, var_variance, var_stddev = \
np.mean(yVal), np.median(yVal),mode(yVal), np.var(yVal),np.std(yVal)
print("Mean: %s\nMedian: %s\nMode: %s\nVariance: %s\nstd dev: %s" 
      %(str(var_mean), str(var_median), str(var_mode), str(var_variance), str(var_stddev)))

# percentile calculation
var_quantile = np.percentile(yVal,np.arange(100))
quantiles = mstats.mquantiles(yVal)


#plot normal distribution 
plt.figure()
#plt.plot(norm.pdf(var_quantile,np.mean(var_quantile), np.std(var_quantile)))
'''for i, q in enumerate(quantiles):
    labels = ['25%', '50%', '75%']
    plt.plot(q, label = labels[i])
    '''
plt.boxplot(yVal)
plt.show()



plt.figure(1, figsize=[20,8])
plt.bar(np.arange(np.size(var_quantile)),np.sort(var_quantile))



def findVariableFreq(dataframe, ixColName,schColName, varName):
    varVal = dataframe.loc[dataframe[ixColName] == varName, schColName]
    varVal = varVal.dropna()
    freq =  varVal.value_counts(sort=True, ascending=False)
    return freq.values, freq.index.values
    
def plotFrequencies_Bar(fignum, x, y, xheader):
    fig = plt.figure(fignum, figsize=[70,8])
    ax = fig.add_subplot(111)
    ax.set_xlim(-1,350)
    ax.bar(x,y)
    plt.xticks(x,xheader)
    plt.setp(ax.get_xticklabels(), fontsize=14, rotation='vertical')
    plt.show()
    

def plotFrequencies_scatter(fignum, x, y, xheader):
    fig = plt.figure(fignum, figsize=[70,8])
    ax = fig.add_subplot(111)
    ax.set_xlim(-1,350)
    ax.scatter(x,y)
    plt.xticks(x,xheader)
    plt.setp(ax.get_xticklabels(), fontsize=14, rotation='vertical')
    plt.show()
    
    
a, b  = findVariableFreq(column, "Entity_Key", "Entity_Value", "app")
#b = b.reshape(np.size(b),1)
a = a.reshape(np.size(a),1)
y = np.arange(np.size(b))


plotFrequencies_Bar(2,xVal,yVal,xHeaders)
plotFrequencies_Bar(3,y,a,b)
