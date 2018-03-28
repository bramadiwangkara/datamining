# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 21:45:11 2018

@author: Akhmad Bakhrul Ilmi
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

d = pd.read_csv("./dataset_noise.csv", index_col=0)
dataset = d.values
#
plt.scatter(dataset[:, 0], dataset[:, 1], marker='o')
plt.show()

nclust = 10

#Proses K-Means
dataset = d.values
kmeans = KMeans(n_clusters=nclust).fit(dataset)
labels = kmeans.labels_

#plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c=labels)
#plt.show()

#datasetBaru = dataset
# Proses Penghapusan data
for i in range(0, nclust):
    count = np.count_nonzero(labels == i)
    if count <= 3:
        indexDelete = np.where(labels == i)
        dataset = np.delete(dataset, indexDelete, axis=0)
        labels = np.delete(labels, indexDelete, axis=0)

kmeans_after = KMeans(n_clusters=3).fit(dataset)
labels_after = kmeans_after.labels_
        
#plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c=labels_after)
#plt.show()
