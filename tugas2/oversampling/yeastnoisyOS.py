import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from collections import Counter


X_data = np.genfromtxt("yeast.dat", delimiter=",", skip_header=13)[:,:-1]
target = np.genfromtxt("yeast.dat", delimiter=",", skip_header=13, usecols=-1, dtype=str)

# print(X_data)

print("raw class : ", Counter(target))

X_train, X_test, y_train, y_test = train_test_split(X_data, target, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("raw accuracy " , metrics.accuracy_score(y_test, y_pred_class))

nclust = 30
dataset = X_data
kmeans = KMeans(n_clusters=nclust).fit(dataset)
labels = kmeans.labels_


for i in range(0, nclust):
    count = np.count_nonzero(labels == i)
    if count <= 3:
        indexDelete = np.where(labels == i)
        dataset = np.delete(dataset, indexDelete, axis=0)
        labels = np.delete(labels, indexDelete, axis=0)

kmeans_after = KMeans(n_clusters=10).fit(dataset)
labels_after = kmeans_after.labels_

#print(dataset)

print("raw class : ", Counter(labels_after))

X_train, X_test, y_train, y_test = train_test_split(dataset, labels_after, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("reduce kmeans accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')
# print (X_sklearn)

#Smote
sm = SMOTE(ratio = 'auto', random_state=0)
new_data1, new_target1 = sm.fit_sample(dataset, labels_after)
print (new_data1.shape)
# print (np.count_nonzero(new_target1=='0'))
# print (np.count_nonzero(new_target1=='1'))

c = Counter(new_target1)
print("class", c) 

X_train, X_test, y_train, y_test = train_test_split(new_data1, new_target1, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("after SMOTE accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')


#tomeklinks
tl = TomekLinks(random_state = 0, ratio = 'not minority', return_indices=True)
new_data2, new_target2, idx_new = tl.fit_sample(new_data1,new_target1)

c = Counter(new_target1)
print("class tomeklinks", c) 

X_train, X_test, y_train, y_test = train_test_split(new_data2, new_target2, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("after cleaning data (SMOTETomeks) accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')




enn = EditedNearestNeighbours(random_state = 0, ratio = 'not minority', return_indices=True)
new_data_ENN, new_target_ENN, idx_new = enn.fit_sample(new_data1,new_target1)

c = Counter(new_target_ENN)
print("class (SMOTEENN) ", c) 

X_train, X_test, y_train, y_test = train_test_split(new_data_ENN, new_target_ENN, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("after cleaning data accuracy (SMOTEENN) " , metrics.accuracy_score(y_test, y_pred_class), '\n')