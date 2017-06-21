# DataAnalysisProject
## Anapy -- A simple package for data analysis, visualization and machine learning
### Sub Packages

> 1. **datamanip**
>> #### Modules
>>> * **CentralValues**: Returns a dictionary containing central values Mean, Median, Range, Variance, Standard Deviation and Quantile
>>> * **DataOps**: Functions to  work with pandas dataframes-- (1)_dataDep_csv(infile)_ returns a deduplicated dataframe; (2) _dataFrameSplit(dataframe, no of records)_ splits a dataframe based on no of records needed
>>> * **datasetSeparator**: Useful for looking into pandas dataframe and do column manipulation like removal of columns, current functions -- (1)_displayCols(dataframe)_ to display columns,(2) _remCols(dataframe)_ to remove columns, (3) _sep_data_target(dataframe)_ to separate data and target
> 2. **externals**
>> #### Modules
>>> * **LoadDataset**: (1)_load_pickle(filestr)_ Loads a pickled object, (2) _data_target_separator(numpy array)_ data, target separator for numpy dataset. Assumes the last column contains the labels.
> 3. **mlops**
>> #### Modules
>>> * **learnfromsample**: An important module which takes training set and test set as inputs with parameters such as sample size, sample methods and classifier. Scaler is optional. Returns test set true labels and predicted labels, training set true labels and predicted labels and Fitting time and Prediction time of the model under examination.
>>>* **Visualization**: Dimensionality reduction in order to visualize dataset in 2D and 3D spaces.
> 4. **sampling**
>>#### Modules
>> This contains different probability based sampling modules - works with numpy datasets
>>> **ClusterSampling**
>>> **RandomSampling**
>>> **StratifiedSampling**
>>> **SystematicSampling
> 5. **misc**
>> This contains very project specific modules which work only with this project scenario
