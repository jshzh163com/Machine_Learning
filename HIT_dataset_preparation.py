# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:51:18 2023

@author: zhua079
"""

import numpy as np

file_path = r'D:\\Downloads\\HIT_dataset\\'
file_name = 'data1.npy'
file = file_path + file_name
f1 = np.load(file)

data1 = f1[:, 0: 6, :].reshape(-1, 20480)
aa = list(data1)

from scipy.io import savemat
file_path_new = r'D:\\Downloads\\HIT_dataset\\data5.mat'
savemat(file_path_new, {'data': data1})

from scipy.io import loadmat
data1 = np.array(list(loadmat(r'D:\Downloads\HIT_dataset\1_healthy.mat').items()), dtype=object)[3, -1]
data_num = int(data1.shape[0]/ 6)
data = []
label = []
for iter in range (data_num):
    row1 = 0 + (iter)* 6
    row2 = 6 + (iter)* 6
    data.append(data1[row1: row2, :])
    label.append(0)
    
    
    
