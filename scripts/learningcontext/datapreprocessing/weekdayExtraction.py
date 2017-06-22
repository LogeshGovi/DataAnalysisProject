# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:15:34 2017

@author: Logesh Govindarajulu
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
#import matplotlib.backends.backend_pdf as bpdf
plt.ioff()


df = pd.read_csv("D:\\lcif\\16032017-IndividualFiles\\FJoinTSDiscreteValuesFilled.csv",
                  dtype='unicode')
ts = pd.to_datetime(df['Timestamp'])
df['weekday'] = ts.dt.weekday
df['hour'] = ts.dt.hour
df['minute'] = ts.dt.minute
le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
le1.fit(df['app_class_label'].dropna().drop_duplicates().values)
class_label = le1.transform(df['app_class_label'].values)
le2.fit(df['app'].dropna().drop_duplicates().values)
app_label = le2.transform(df['app'].values)
df['label'] = pd.DataFrame(class_label)
df['appcat'] = pd.DataFrame(app_label)
df1 = df.drop(['PK','EventId','Timestamp'], axis=1)
df1.to_csv("D:\\lcif\\16032017-IndividualFiles\\prelimMLdataset.csv", sep=',', header=True, index=False)