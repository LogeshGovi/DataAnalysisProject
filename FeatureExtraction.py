# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:04:28 2017
@author: Simone
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
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

xVal = xVal + 1

records = np.concatenate((xhead,yVal),axis=1)
records_sorted = records[records[:,1].argsort()[::-1]]

"""
print(records_sorted)
fig = plt.figure(1, [40, 8])
fig.clf()
ax = fig.add_subplot(111)
# Set the x-axis limit
ax.set_xlim(-1,50)
plt.xticks(xVal,xHeaders)
# Change of fontsize and angle of xticklabels
plt.setp(ax.get_xticklabels(), fontsize=14, rotation='vertical')
ax.bar(xVal,yVal)
"""
def findVariableFreq(dataframe, ixColName,schColName, varName):
    varVal = dataframe.loc[dataframe[ixColName] == varName, schColName]
    varVal = varVal.dropna()
    freq =  varVal.value_counts(sort=True, ascending=False)
    return freq.values, freq.index.values
    
def plotFrequencies_Bar(fignum, x, y, xheader):
    fig = plt.figure(figsize=[70,8])
    ax = fig.add_subplot(111)
    ax.set_xlim(-1,350)
    ax.bar(x,y)
    plt.xticks(x,xheader)
    plt.setp(ax.get_xticklabels(), fontsize=14, rotation='vertical')
    plt.show()
    
    
a, b  = findVariableFreq(column, "Entity_Key", "Entity_Value", "app")
#b = b.reshape(np.size(b),1)
a = a.reshape(np.size(a),1)
y = np.arange(np.size(b))
plotFrequencies_Bar(2,y,a,b)
