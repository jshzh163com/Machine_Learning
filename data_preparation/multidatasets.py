# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:36:29 2024

@author: zhua079
"""

# coding=utf-8

import os
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm


# Digital data was collected at 12,000 samples per second
f_size, t_size = 93, 93
work_condition = ['_F.mat', '_N.mat']
dataname = {0: [os.path.join('CWRU'+work_condition[0]),
                os.path.join('CWRU'+work_condition[0])],
            1: [os.path.join('HUST'+work_condition[0]),
                os.path.join('HUST'+work_condition[0])],
            2: [os.path.join('UODS'+work_condition[0]),
                os.path.join('UODS'+work_condition[0])],
            3: [os.path.join('XJTU'+work_condition[0]),
                os.path.join('XJTU'+work_condition[0])],
            }  # 1730rpm


label = [i for i in range(0, 2)]


def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n == 0:
                path1 = os.path.join(root, dataname[N[k]][n])
            else:
                path1 = os.path.join(root, dataname[N[k]][n])
            data1, lab1 = data_load(path1, label=label[n])
            data += data1
            lab += lab1

    return [data, lab]


def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    # datanumber = axisname.split(".")
    fl = np.array(list(loadmat(filename).items()), dtype=object)[
        3, -1]

    data = []
    lab = []
    start, end = 0, t_size
    while end <= t_size*400:
        data.append(fl[:, start: end])
        lab.append(label)
        start += t_size
        end += t_size
    return data, lab
