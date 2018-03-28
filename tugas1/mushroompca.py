import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA as sklearnPCA

headers = ["cap_shape","cap_surface","cap_color","bruises","odor","gill_attachment","gill_spacing","gill_size",
           "gill_color","stalk_shape","stalk_root","stalk_surface_above_ring","stalk_surface_below_ring",
           "stalk_color_above_ring","stalk_color_below_ring","veil_type","veil_color",
           "ring_number","ring_type","spore_print_color","population","habitat","classification"]

data = pd.read_csv('hasil.csv',header=None, names=headers, na_values="?" )

res = data['stalk_root'].value_counts().index.tolist()
data = data.fillna({"stalk_root": res[0]})

data[data.isnull().any(axis=1)]

target = 'classification' # The class we want to predict
labels = data[target]

features = data.drop(target, axis=1) # Remove the target class from the dataset
categorical = features.columns # Since every fearure is categorical we use features.columns
features = pd.concat([features, pd.get_dummies(features[categorical])], axis=1) # Convert every categorical feature with one hot encoding
features.drop(categorical, axis=1, inplace=True) # Drop the original feature, leave only the encoded ones

labels = pd.get_dummies(labels)['p']

X_std = StandardScaler().fit_transform(features)
y = labels

sklearn_pca = sklearnPCA(n_components=2)
X_sklearn = sklearn_pca.fit_transform(X_std)

plt.scatter(X_sklearn[:, 0], X_sklearn[:, 1], marker='o', c=y)
plt.show()
