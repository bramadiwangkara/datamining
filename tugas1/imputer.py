import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

#dataset = pd.read_csv('mushroom.csv')

dataset=open("mushroom.csv","r")
lines = dataset.readlines()
result=[]
for x in lines:
    result.append(x.split(',')[10])
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 23].values
dataset.close()


for x, res in enumerate(result):
	if (res == 'b'):
		result[x] = 1
	elif (res == 'c'):
		result[x] = 2
	elif (res == 'u'):
		result[x] = 3
	elif (res == 'e'):
		result[x] = 4
	elif (res == 'z'):
		result[x] = 5
	elif (res == 'r'):
		result[x] = 6
	else:
		result[x] = 0

#print(result)

imputer = Imputer(missing_values = 0, strategy = 'most_frequent', axis = 0)

imputer = imputer.fit(result[:1])

#X[:, 1:23] = imputer.transform(X[:, 1:23])

#index = ['Cap-shape', 'Cap-surface', 'Cap-color', 'Bruises', 'Odor', 'Gill-attachment', 'Gill-spacing', 'Gill-size',' Gill-color', 'Stalk-shape', 'Stalk-root', 'Stalk-surface-above-ring', 'Stalk-surface-below-ring', 'Stalk-color-above-ring', 'Stalk-color-below-ring', 'Veil-type', 'Veil-color', 'Ring-number', 'Ring-type', 'Spore-print-color', 'Population', 'Habitat']
#df = pd.DataFrame(X, columns=index)
#print(df)
