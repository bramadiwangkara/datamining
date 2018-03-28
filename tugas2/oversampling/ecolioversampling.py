import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import classification_report as cr


X_data = np.genfromtxt("ecoli4.dat", delimiter=",", skip_header=12)[:,:-1]
target = np.genfromtxt("ecoli4.dat", delimiter=",", skip_header=12, usecols=-1, dtype=str)

#print(X_data);
#print(target);

c = Counter(target)
print(c) 

X_train, X_test, y_train, y_test = train_test_split(X_data, target, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("raw accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')

print(cr(y_test, y_pred_class), '\n')


# target[target == 'tested_positive'] = 0
# target[target == 'tested_negative'] = 1
# target = list(map(int, target))

#X_std = StandardScaler().fit_transform(X_data)
#sklearn_pca = sklearnPCA(n_components=2)
#X_data = sklearn_pca.fit_transform(X_std)

#origindata

# for x, res in enumerate(target):
# 	if (res == ' positive'):
# 		target[x] = 1
# 	elif(res == ' negative') :
# 		target[x] = 0

# noisy data
# nclust = 20
# dataset = X_data
# kmeans = KMeans(n_clusters=nclust).fit(dataset)
# labels = kmeans.labels_


# for i in range(0, nclust):
#     count = np.count_nonzero(labels == i)
#     if count <= 3:
#         indexDelete = np.where(labels == i)
#         dataset = np.delete(dataset, indexDelete, axis=0)
#         labels = np.delete(labels, indexDelete, axis=0)

# kmeans_after = KMeans(n_clusters=2).fit(dataset)
# labels_after = kmeans_after.labels_

# c = Counter(labels_after)
# print(c) 

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# Y_pd = pd.DataFrame(X_data);
# x,y = Y_pd[0], Y_pd[1];
# ax1.scatter(x, y, c=target,marker='+');
# ax1.set_title('raw data')

# Y_pd = pd.DataFrame(dataset);
# x,y = Y_pd[0], Y_pd[1];
# ax2.scatter(x, y, c=labels_after,marker='+');
# ax2.set_title('after noisy data')

#Smote
sm = SMOTE(ratio = 'auto', random_state=42)
new_data1, new_target1 = sm.fit_sample(X_data, target)
# print (new_data1.shape)
c = Counter(new_target1)
print("class after Smote ", c)
# print (np.count_nonzero(new_target1=='0'))
# print (np.count_nonzero(new_target1=='1'))

#adasyn
ada = ADASYN(ratio = 'auto', random_state=42)
new_data_ad, new_target_ad = ada.fit_sample(X_data, target)
# print (new_data_ad.shape)
# print (np.count_nonzero(new_target_ad=='0'))
# print (np.count_nonzero(new_target_ad=='1'))

#accuracy smote
X_train, X_test, y_train, y_test = train_test_split(new_data1, new_target1, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("after SMOTE accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')
print(cr(y_test, y_pred_class), '\n')


c= Counter(new_target_ad)
print("class after ADASYN ", c)

#accuracy adasyn
X_train, X_test, y_train, y_test = train_test_split(new_data_ad, new_target_ad, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("after ADASYN accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')
print(cr(y_test, y_pred_class), '\n')

# Y_pd = pd.DataFrame(new_data1);
# x,y = Y_pd[0], Y_pd[1];
# ax2.scatter(x, y, c=new_target1,marker='+');
# ax2.set_title('smote')

# Y_pd = pd.DataFrame(new_data_ad);
# x,y = Y_pd[0], Y_pd[1];
# ax3.scatter(x, y, c=new_target_ad,marker='+');
# ax3.set_title('adasyn')

#Tomek Links
tl = TomekLinks(random_state = 0, ratio = 'not minority', return_indices=True)
new_data2, new_target2, idx_new = tl.fit_sample(new_data1,new_target1)
#print (new_data2.shape)
c= Counter(new_target2)
print("Class after data cleaning (SMOTETomeks) ", c)

X_train, X_test, y_train, y_test = train_test_split(new_data2, new_target2, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("after SMOTETomeks accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')
print(cr(y_test, y_pred_class), '\n')
#print (np.count_nonzero(new_target2=='0'))
#print (np.count_nonzero(new_target2=='1'))
#
#Y_pd = pd.DataFrame(new_data2);
#x,y = Y_pd[0], Y_pd[1];
#ax3.scatter(x, y, c=new_target2,marker='+');
#ax3.set_title('smote + tomek links')

enn = EditedNearestNeighbours(random_state = 0, ratio = 'not minority', return_indices=True)
new_data_ENN, new_target_ENN, idx_new = enn.fit_sample(new_data1,new_target1)

c= Counter(new_target_ENN)
print("Class after data cleaning (SMOTEENN) ", c)
X_train, X_test, y_train, y_test = train_test_split(new_data_ENN, new_target_ENN, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("after SMOTEENN accuracy " , metrics.accuracy_score(y_test, y_pred_class))
print(cr(y_test, y_pred_class), '\n')
# fig.tight_layout()
# plt.show()
