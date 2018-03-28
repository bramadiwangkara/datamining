from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss 
from imblearn.under_sampling import OneSidedSelection 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


X_data = np.genfromtxt("ecoli4.dat", delimiter=",", skip_header=12)[:,:-1]
target = np.genfromtxt("ecoli4.dat", delimiter=",", skip_header=12, usecols=-1, dtype=str)

print('Raw Data', Counter(target), '\n')


for x, res in enumerate(target):
	if (res == ' positive'):
		target[x] = 1
	elif(res == ' negative') :
		target[x] = 0
# X_train, X_test, y_train, y_test = train_test_split(X_data, target, random_state=0)
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# y_pred_class = logreg.predict(X_test)
# print("raw accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
Y_pd = pd.DataFrame(X_data);
x,y = Y_pd[0], Y_pd[1];
ax1.scatter(x, y, c=target,marker='+');
ax1.set_title('raw data')

nm = NearMiss(random_state=0)
new_data_KNN, new_target_KNN = nm.fit_sample(X_data, target)
print('Resampled KNN dataset shape {}'.format(Counter(new_target_KNN)))

Y_pd = pd.DataFrame(new_data_KNN);
x,y = Y_pd[0], Y_pd[1];
ax2.scatter(x, y, c=new_target_KNN,marker='+');
ax2.set_title('KNN data')

oss = OneSidedSelection(random_state=0)
new_data_OSS, new_target_OSS = oss.fit_sample(X_data, target)
print('Resampled OSS dataset shape {}'.format(Counter(new_target_OSS)))


Y_pd = pd.DataFrame(new_data_OSS);
x,y = Y_pd[0], Y_pd[1];
ax3.scatter(x, y, c=new_target_OSS,marker='+');
ax3.set_title('OSS data')
# X_train, X_test, y_train, y_test = train_test_split(new_data_KNN, new_target_KNN, random_state=0)
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# y_pred_class = logreg.predict(X_test)
# print("KNN under_sampling accuracy " , metrics.accuracy_score(y_test, y_pred_class), '\n')
fig.tight_layout()
plt.show()
