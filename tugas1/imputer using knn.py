# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 07:22:57 2018

@author: Akhmad Bakhrul Ilmi
"""

import knn_impute as ki
import pandas as pd

df = pd.read_csv('mushroom.csv')
print(df)
new_data = ki.knn_impute(target=df['Age'], attributes=df.drop(['Age', 'Purchased'], 1),
                                    aggregation_method="mean", k_neighbors=2, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)
df = df.assign(Age = new_data)
print(df)

new_data = ki.knn_impute(target=df['Salary'], attributes=df.drop(['Age', 'Purchased'], 1),
                                    aggregation_method="mean", k_neighbors=1, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)

df = df.assign(Salary = new_data)
print(df)