import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import csv


dataset = pd.read_csv('mushroom.csv', skiprows=0, header=None)

#mv to nan
for column in dataset:
    dataset[column] = dataset[column].replace('?', np.NaN)
#nan to mode

#print (dataset)

mode=[0]*23;
for i in range(23):
    mode[i]=dataset[i].mode().ix[0]
for i in range(23):
    dataset[i].fillna(mode[i], axis=0, inplace=True)

#listing class
cls = dataset[22].tolist()
u, clr = np.unique(cls, return_inverse=True)
#print clr

#print ds
#print (dataset)
X = dataset.iloc[:,:23].values
with open('hasil2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(X)
