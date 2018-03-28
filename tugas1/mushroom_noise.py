import pandas as pd
import numpy as np
from sklearn import preprocessing
from kmodes.kmodes import KModes

headers = ["cap_shape","cap_surface","cap_color","bruises","odor","gill_attachment","gill_spacing","gill_size",
           "gill_color","stalk_shape","stalk_root","stalk_surface_above_ring","stalk_surface_below_ring",
           "stalk_color_above_ring","stalk_color_below_ring","veil_type","veil_color",
           "ring_number","ring_type","spore_print_color","population","habitat","classification"]

data = pd.read_csv('hasil.csv',header=None, names=headers, na_values="?" )
data.head()

res = data['stalk_root'].value_counts().index.tolist()
data = data.fillna({"stalk_root": res[0]})

data[data.isnull().any(axis=1)]

nclust = 25	#jumlah centroid
km = KModes(n_clusters=nclust, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(data)
labels = km.labels_

dataset = data.values
for i in range(0, nclust):
    count = np.count_nonzero(labels == i)
    if count <= 100:	#kelas yang anggotanya kurang dari 100 dihapus
        indexDelete = np.where(labels == i)
        dataset = np.delete(dataset, indexDelete, axis=0)
        labels = np.delete(labels, indexDelete, axis=0)

df = pd.DataFrame(dataset)
print(df)
df.to_csv('reduksi_noise.csv',header=None,index=None, sep=',')
