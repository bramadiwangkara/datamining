import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

X_data = np.genfromtxt("syn.csv", delimiter=",", skip_header=1)[:,:-1]
target = np.genfromtxt("syn.csv", delimiter=",", skip_header=1, usecols=-1, dtype=str)

#print(X_data);
#print(target);

# target[target == 'tested_positive'] = 0
# target[target == 'tested_negative'] = 1
# target = list(map(int, target))

#X_std = StandardScaler().fit_transform(X_data)
#sklearn_pca = sklearnPCA(n_components=2)
#X_data = sklearn_pca.fit_transform(X_std)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
Y_pd = pd.DataFrame(X_data);
x,y = Y_pd[0], Y_pd[1];
ax1.scatter(x, y, c=target,marker='+');
ax1.set_title('data')

#Smote
sm = SMOTE(ratio = 'auto', random_state=0)
new_data1, new_target1 = sm.fit_sample(X_data, target)
print (new_data1.shape)
print (np.count_nonzero(new_target1=='0'))
print (np.count_nonzero(new_target1=='1'))

#adasyn
ada = ADASYN(ratio = 'auto', random_state=0)
new_data_ad, new_target_ad = ada.fit_sample(X_data, target)
print (new_data_ad.shape)
print (np.count_nonzero(new_target_ad=='0'))
print (np.count_nonzero(new_target_ad=='1'))


Y_pd = pd.DataFrame(new_data1);
x,y = Y_pd[0], Y_pd[1];
ax2.scatter(x, y, c=new_target1,marker='+');
ax2.set_title('smote')

Y_pd = pd.DataFrame(new_data_ad);
x,y = Y_pd[0], Y_pd[1];
ax3.scatter(x, y, c=new_target_ad,marker='+');
ax3.set_title('adasyn')

#Tomek Links
#tl = TomekLinks(random_state = 0, ratio = 'not minority', return_indices=True)
#new_data2, new_target2, idx_new = tl.fit_sample(new_data1,new_target1)
#print (new_data2.shape)
#print (np.count_nonzero(new_target2=='0'))
#print (np.count_nonzero(new_target2=='1'))
#
#Y_pd = pd.DataFrame(new_data2);
#x,y = Y_pd[0], Y_pd[1];
#ax3.scatter(x, y, c=new_target2,marker='+');
#ax3.set_title('smote + tomek links')

fig.tight_layout()
plt.show()