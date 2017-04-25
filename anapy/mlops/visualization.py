# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:29:07 2017

@author: Logesh Govindarajulu
"""

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pylab

class Visualization:
    def __init__(self, data, target):
        # data and target must be numpy arrays
        self.data = data
        self.target = target

    def dim_red(self, transform , n_dim = 2):
        trans = transform
        #data = StandardScaler.fit_transform(self.data)

        if trans == 'FactorAnalysis':
            factor_analysis = FactorAnalysis(n_components=n_dim)
            t_data = factor_analysis.fit_transform(data)
            return t_data

        elif trans == 'FastICA':
            fastICA = FastICA(n_components=n_dim)
            t_data = fastICA.fit_transform(data)
            return t_data

        elif trans == 'PCA':
            pca = PCA(n_components=n_dim)
            t_data = pca.fit_transform(self.data)
            return t_data

        elif trans == 'Isomap':
            isomap = Isomap(n_neighbors=5,n_components=n_dim)
            t_data = isomap.fit_transform(data)
            return t_data

        elif trans == 'MDS':
            mds = MDS(n_components=n_dim)
            t_data = mds.fit_transform(data)
            return t_data

        elif trans == 'LinearDiscriminantAnalysis':
            lda = LinearDiscriminantAnalysis(n_components=n_dim)
            t_data = lda.fit_transform(data)
            return t_data

        else:
            print("The dimensional reduction algorithm" + trans+ "is not supported")
            return None

    def visualizedata(self,t_data):
        noOfColors = np.size(np.unique(self.target))
        colormap = pylab.get_cmap('gist_rainbow')
        colors = (colormap(1.*i/noOfColors) for i in range(noOfColors))
        colmap = ListedColormap(colormap)
        if np.shape(t_data)[1]==2:
            fig = plt.figure(figsize=(20,10))
            plots = plt.scatter(t_data[:,0],t_data[:,1],c=self.target,cmap=colmap)
            plt.show()
            return 0
        else:
            print("The dimensions given for plotting is more than 2")




