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
from mpl_toolkits.mplot3d import Axes3D

class Visualization:
    """
    This class handles two operations namely, dimensionality reduction and plotting of the data

    Functions Provided:
    (1) dim_red - Dimensionality reduction
    Supported Algorithms: FactorAnalysis, FastICA, PCA, Isomap, MDS, LinearDiscriminantAnalysis

    (2) visualizedata - Uses matplotlib to create a scatter plot showing the data points
    only plotting of two dimensional data is supported

    """
    def __init__(self, data, target):
        # data and target must be numpy arrays
        self.data = data
        self.target = target

    def dim_red(self, transform , n_dim = 2):
        trans = transform
        #data = StandardScaler.fit_transform(self.data)

        if trans == 'FactorAnalysis':
            factor_analysis = FactorAnalysis(n_components=n_dim)
            t_data = factor_analysis.fit_transform(self.data)
            return t_data, trans

        elif trans == 'FastICA':
            fastICA = FastICA(n_components=n_dim)
            t_data = fastICA.fit_transform(self.data)
            return t_data, trans

        elif trans == 'PCA':
            pca = PCA(n_components=n_dim)
            t_data = pca.fit_transform(self.data)
            return t_data, trans

        elif trans == 'Isomap':
            isomap = Isomap(n_neighbors=5,n_components=n_dim)
            t_data = isomap.fit_transform(self.data)
            return t_data, trans

        elif trans == 'MDS':
            mds = MDS(n_components=n_dim)
            t_data = mds.fit_transform(self.data)
            return t_data, trans

        elif trans == 'LinearDiscriminantAnalysis':
            lda = LinearDiscriminantAnalysis(n_components=n_dim)
            t_data = lda.fit_transform(self.data, np.ravel(self.target))
            return t_data, trans

        else:
            print("The dimensional reduction algorithm" + trans+ "is not supported")
            return None

    def visualizedata(self,t_data,trans):
        noOfColors = np.size(np.unique(self.target))
        color_list = ListedColormap(
                                    [
                                      "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
                                      "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
                                      "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
                                      "#61615A"
                                    ]
                                   )
        if np.shape(t_data)[1]==2:
            fig = plt.figure(1, figsize=(30,15))
            plots = plt.scatter(t_data[:,0],t_data[:,1],s=20,c=self.target,cmap=color_list,edgecolors='face',marker='_')
            plt.title("Dimensionality Reduction Method: " + trans)
            plt.show()
            return 0
        else:
            print("This method plots only two dimensional data")

    def visualizedata3d(self,t_data, trans):
        colours = [
                      "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
                      "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
                      "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
                      "#61615A"

                  ]
        noOfColors = np.size(np.unique(self.target))
        color_list = ListedColormap(colours)
        if np.shape(t_data)[1]==3:
            fig = plt.figure(figsize=(30,15))
            ax = fig.add_subplot(111, projection='3d')
            plots = ax.scatter(t_data[:,0],t_data[:,1],t_data[:,2],s=20 ,c=self.target,cmap=color_list,edgecolors='face',marker='_')
            plt.title("Dimensionality Reduction Method: " + trans)
            #for angle in range(0,12):
                #ax.view_init(30,30*(angle+1))
                #plt.draw()
                #plt.pause(.1)
            plt.show()
            return 0
        else:
            print("This method plots only three dimensional data")

