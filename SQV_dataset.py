# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 23:26:46 2023

@author: zhua079
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns
# sns.set(style='darkgrid')
# mpl.rcParams['font.sans-serif'] = ['SimHei']


def txt_read(filename):
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    rows = len(content)                                 # 文件行数
    datamat = np.zeros((rows - 16, 1))                  # 初始化数组
    row_count = 0

    for i in range(16, rows):                           # 从17行开始读取，文件所决定
        content[i] = content[i].strip().split('\t')
        datamat[row_count, 0] = content[i][1]           # 获取第i行2列数据
        row_count += 1

    file.close()
    return datamat


def get_speed(speed_seq, Fs=25600, threshold=1):
    r"""
    speed_seq: 转速脉冲信号
    Fs:        采频
    threshold: 阈值设定
    """
    x_normal = np.linspace(0, len(speed_seq) / Fs,
                           len(speed_seq))              # Times
    Temp = []
    for l1 in range(0, len(speed_seq)):
        if (speed_seq[l1] >= threshold and speed_seq[l1+1] <= threshold):
            Temp.append(x_normal[l1])
    Speed = []
    Time = []
    for l1 in range(0, len(Temp)):
        if (l1 % 1 == 0 and l1 != 0):
            Speed.append(1 / (Temp[l1] - Temp[l1 - 1])
                         * 60)                      # Speed
            # Time
            Time.append(Temp[l1])
    return Speed, Time


speed_file = r'D:\Downloads\SQV-public\IF_1\REC3606_ch3.txt'

rawdata_speed = txt_read(speed_file).squeeze(
)                          # 读取转速脉冲信号
rawdata_speed = rawdata_speed * -1                                      # 转速脉冲信号转正
# 提取转速曲线
speed, time = get_speed(rawdata_speed)

t = np.arange(len(rawdata_speed))/25600.

fig, ax = plt.subplots(2, 1, figsize=(6, 4))
TICKSIZE = 15
TITLESIZE = 17
plt.subplots_adjust(wspace=0.2, hspace=0.7)

axes = ax.flatten()
axes[0].plot(t, rawdata_speed, 'r-', lw=0.5)
axes[1].plot(time, speed, 'r-', lw=0.5)

[axes[i].tick_params(labelsize=TICKSIZE) for i in range(2)]
[axes[i].set_xlabel('Time (s)', fontsize=TICKSIZE) for i in range(2)]
axes[0].set_ylabel('Voltage (V)', fontsize=TICKSIZE)
axes[0].set_title('转速脉冲', fontsize=TITLESIZE)
axes[1].set_ylabel('Speed (rpm)', fontsize=TICKSIZE)
axes[1].set_title('转速曲线', fontsize=TITLESIZE)

plt.savefig('./Speed_curve.png', bbox_inches='tight', dpi=800)
# plt.show()
