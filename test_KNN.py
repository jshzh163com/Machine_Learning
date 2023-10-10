# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:52:12 2023

@author: zhua079
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from CWRUdata import get_files

#   file path
data_dir = r'D:\Downloads\Mechanical-datasets-master\dataset'
list_data1 = get_files(data_dir, [0])

#   flatten the signals, shown in a row
signal_size = 512
data = np.array(list_data1[0]).reshape(-1, signal_size)
label = np.array(list_data1[1])

#   data split
data_train, data_test, label_train, label_test = train_test_split(
    data, label, test_size=0.3, shuffle=True)

knn = KNN(n_neighbors=10, algorithm='auto', 
          weights='distance',leaf_size=30, 
          metric='minkowski', p=2,
          metric_params=None, n_jobs=1)

#   original data
train_knn = knn.fit(data_train, label_train)
# train_knn = knn.fit(data, label)
#   validate
train_knn.score(data_test, label_test)
# print(knn.predict(data_test))
# train_knn.score(data_train, label_train)

'''
combined with PCA 
'''

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Data = StandardScaler().fit_transform(data)
pca = PCA(n_components = 10)
data_pca = pca.fit_transform(Data)

data_pca_train, data_pca_test, label_pca_train, label_pca_test = train_test_split(
    data_pca, label, test_size=0.1, shuffle=True)

#   data processed by PCA
train_knn_pca = knn.fit(data_pca_train, label_pca_train)
train_knn_pca.score(data_pca_test, label_pca_test)




