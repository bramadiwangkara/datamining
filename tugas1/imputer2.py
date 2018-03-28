import random
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('mushroom.csv')
X = dataset.iloc[:,:23].values
#print(X)
#y = dataset.iloc[:, 23].values

result=[] 
result = X[:, 10:11]
#print (result)
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


#print(X)
imputer = Imputer(missing_values = 0, strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 10:11])
X[:, 10:11] = imputer.transform(X[:, 10:11])
print(X[:, 22:23])
#print(result)

for x, res in enumerate(result):
	if (res == 1):
		result[x] = 'b'
	elif (res == 2):
		result[x] = 'c'
	elif (res == 3):
		result[x] = 'u'
	elif (res == 4):
		result[x] = 'e'
	elif (res == 5):
		result[x] = 'z'
	elif (res == 6):
		result[x] = 'r'

#print(result)

with open('hasil.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(X[:, 0:23])
#print(dataset) 
#index = ['Cap-shape', 'Cap-surface', 'Cap-color', 'Bruises', 'Odor', 'Gill-attachment', 'Gill-spacing', 'Gill-size',' Gill-color', 'Stalk-shape', 'Stalk-root', 'Stalk-surface-above-ring', 'Stalk-surface-below-ring', 'Stalk-color-above-ring', 'Stalk-color-below-ring', 'Veil-type', 'Veil-color', 'Ring-number', 'Ring-type', 'Spore-print-color', 'Population', 'Habitat']
#df = pd.DataFrame(X, columns=index)
#print(df)
