import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from collections import Counter
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

#print (y)

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
print("reduce kmeans accuracy " , metrics.accuracy_score(y_test, y_pred_class))
# print (X_sklearn)



# plt.scatter(X_sklearn[:, 0], X_sklearn[:, 1], marker='+', c='b')
# plt.show()
