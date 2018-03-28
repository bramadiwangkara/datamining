from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss 
from imblearn.under_sampling import OneSidedSelection 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


X_data = np.genfromtxt("ecoli4.dat", delimiter=",", skip_header=12)[:,:-1]
target = np.genfromtxt("ecoli4.dat", delimiter=",", skip_header=12, usecols=-1, dtype=str)

print('Raw Data', Counter(target), '\n')

X_train, X_test, y_train, y_test = train_test_split(X_data, target, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("raw accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')


nm = NearMiss(random_state=0)
new_data_KNN, new_target_KNN = nm.fit_sample(X_data, target)
print('Resampled KNN dataset shape {}'.format(Counter(new_target_KNN)), '\n')


oss = OneSidedSelection(random_state=0)
new_data_OSS, new_target_OSS = oss.fit_sample(X_data, target)
print('Resampled OSS dataset shape {}'.format(Counter(new_target_OSS)))


X_train, X_test, y_train, y_test = train_test_split(new_data_KNN, new_target_KNN, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("KNN under_sampling accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')

X_train, X_test, y_train, y_test = train_test_split(new_data_OSS, new_target_OSS, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print("OSS under_sampling accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')