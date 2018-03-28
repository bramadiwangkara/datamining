# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 21:45:11 2018

@author: Akhmad Bakhrul Ilmi
"""
import csv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



d = pd.read_csv("hasil.csv", index_col=0)
dataset = d.iloc[:,:-1].values
X = d.iloc[:,:23].values

print(X[:, 21:22])

for x,strings in enumerate(dataset):
	for y,subs in enumerate(strings):
 		dataset[x,y] = ord(dataset[x,y])
 		#print (ord(dataset[x,y]))
		
	#print (x)

print (X)
#print (dataset[:,21])
#plt.scatter(dataset[:, 0], dataset[:, 1], marker='o')
#plt.show()

nclust = 22

#Proses K-Means
#dataset = d.values
kmeans = KMeans(n_clusters=nclust).fit(dataset)
labels = kmeans.labels_

#plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c=labels)
#plt.show()

#datasetBaru = dataset
# Proses Penghapusan data
for i in range(0, nclust):
   count = np.count_nonzero(labels == i)
   if count <= 22:
       indexDelete = np.where(labels == i)
       dataset = np.delete(dataset, indexDelete, axis=0)
       labels = np.delete(labels, indexDelete, axis=0)

kmeans_after = KMeans(n_clusters=22).fit(dataset)
labels_after = kmeans_after.labels_
        
plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', c=labels_after)
plt.show()
