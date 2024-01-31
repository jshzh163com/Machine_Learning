# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 23:22:48 2024

@author: zhua079
"""

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import random
import torch
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from multidatasets import get_files


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(0)

data_dir = r'D:\Matlab codes\N_F'
list_data1 = get_files(data_dir, [0])

#   flatten the signals, shown in a row
data = np.array(list_data1[0]).reshape(len(list_data1[0]), -1)
label = np.array(list_data1[1])


#   data split
data_train, data_test, label_train, label_test = train_test_split(
    data, label, test_size=0.8, shuffle=True)

knn = KNN(n_neighbors=2, algorithm='auto',
          weights='distance', leaf_size=50,
          metric='minkowski', p=2,
          metric_params=None, n_jobs=1)

train_knn = knn.fit(data_train, label_train)
train_knn.score(data_test, label_test)

set_random_seed(0)
list_data2 = get_files(data_dir, [0])
data_target = np.array(list_data2[0]).reshape(len(list_data1[0]), -1)
label_target = np.array(list_data2[1])
train_knn.score(data_target, label_target)
print(knn.predict(data_target))


'''
combined with PCA 
'''

set_random_seed(0)
Data = StandardScaler().fit_transform(data)
pca = PCA(n_components=10)
data_pca = pca.fit_transform(Data)

data_pca_train, data_pca_test, label_pca_train, label_pca_test = train_test_split(
    data_pca, label, test_size=0.2, shuffle=True)

train_knn_pca = knn.fit(data_pca_train, label_pca_train)
train_knn_pca.score(data_pca_test, label_pca_test)


set_random_seed(0)
list_data2 = get_files(data_dir, [1])
data_target = np.array(list_data2[0]).reshape(len(list_data1[0]), -1)
label_target = np.array(list_data2[1])

Data_t = StandardScaler().fit_transform(data_target)
pca_t = PCA(n_components=10)
data_pca_t = pca.fit_transform(Data_t)

train_knn_pca.score(data_pca_t, label_target)
# print(train_knn_pca.predict(data_pca_t))


'''
manually extracted features
'''


set_random_seed(0)
#   load data
list_data1 = get_files(data_dir, [3])

data = np.array(list_data1[0]).reshape(len(list_data1[0]), -1)
label = np.array(list_data1[1])

para_1 = np.mean(data, 1)
para_2 = np.std(data, 1)
para_3 = np.sqrt(np.mean(data**2, 1))
para_4 = np.array(list(map(max, abs(data))))
para_5 = stats.skew(data, 1)
para_6 = stats.kurtosis(data, 1)
para_7 = np.mean(abs(np.fft.fft(data)), 1)
para_8 = np.std(abs(np.fft.fft(data)), 1)
para_9 = np.sqrt(np.mean(abs(np.fft.fft(data))**2, 1))

para = np.vstack((para_1, para_2, para_3, para_4, para_5,
                 para_6, para_7, para_8, para_9)).T

'''
parameter selection
'''

# input KNN
set_random_seed(0)
para_train, para_test, label_train, label_test = train_test_split(
    para, label, test_size=0.3, shuffle=True)

knn = KNN(n_neighbors=10, algorithm='auto',
          weights='distance', leaf_size=30,
          metric='minkowski', p=2,
          metric_params=None, n_jobs=1)

#   target data
train_knn = knn.fit(para_train, label_train)
train_knn.score(para_test, label_test)

list_data2 = get_files(data_dir, [2])
data = np.array(list_data2[0]).reshape(len(list_data1[0]), -1)
label = np.array(list_data2[1])
para_1 = np.mean(data, 1)
para_2 = np.std(data, 1)
para_3 = np.sqrt(np.mean(data**2, 1))
para_4 = np.array(list(map(max, abs(data))))
para_5 = stats.skew(data, 1)
para_6 = stats.kurtosis(data, 1)
para_7 = np.mean(abs(np.fft.fft(data)), 1)
para_8 = np.std(abs(np.fft.fft(data)), 1)
para_9 = np.sqrt(np.mean(abs(np.fft.fft(data))**2, 1))

para = np.vstack((para_1, para_2, para_3, para_4, para_5,
                 para_6, para_7, para_8, para_9)).T
train_knn.score(para, label)
# print(train_knn.predict(para))


'''
combined with PCA
'''

set_random_seed(0)
Para = StandardScaler().fit_transform(para)
pca = PCA(n_components=6)
para_pca = pca.fit_transform(Para)

para_pca_train, para_pca_test, label_pca_train, label_pca_test = train_test_split(
    para_pca, label, test_size=0.3, shuffle=True)

#   data processed by PCA
train_knn_pca = knn.fit(para_pca_train, label_pca_train)
train_knn_pca.score(para_pca_test, label_pca_test)

list_data2 = get_files(data_dir, [2])
data = np.array(list_data2[0]).reshape(len(list_data1[0]), -1)
label = np.array(list_data1[1])
para_1 = np.mean(data, 1)
para_2 = np.std(data, 1)
para_3 = np.sqrt(np.mean(data**2, 1))
para_4 = np.array(list(map(max, abs(data))))
para_5 = stats.skew(data, 1)
para_6 = stats.kurtosis(data, 1)
para_7 = np.mean(abs(np.fft.fft(data)), 1)
para_8 = np.std(abs(np.fft.fft(data)), 1)
para_9 = np.sqrt(np.mean(abs(np.fft.fft(data))**2, 1))

para = np.vstack((para_1, para_2, para_3, para_4, para_5,
                 para_6, para_7, para_8, para_9)).T
set_random_seed(0)
Para = StandardScaler().fit_transform(para)
pca = PCA(n_components=6)
para_pca = pca.fit_transform(Para)
train_knn_pca.score(para_pca, label)
# print(train_knn_pca.predict(para_pca))
