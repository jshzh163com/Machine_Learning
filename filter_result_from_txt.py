# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 20:59:27 2023

@author: zhua079
"""

import re
import numpy as np


file = open(
    r'C:\Users\zhua079\Desktop\test_no_pretrained_CWRU_30_epochs.txt', 'r')
content = file.read()
file.close()

numbers = content.split(': ')

pattern = '(\d\.\d\d\d)'  # 匹配一个或多个数字


numbers = re.findall(pattern, content)
print(numbers)

a = []
[a.append(numbers[x]) for x in range(90)]

aa = np.array(a).reshape(30, -1)

'''
another
'''

pattern1 = '(val_accuracy: (\d\.\d\d\d))'  # 匹配一个或多个数字

numbers1 = re.findall(pattern1, content)
b = []
[b.append(numbers1[x][1]) for x in range(90)]

a = []
[a.append(numbers[x][0]) for x in range(90)]

aa = np.array(a).reshape(30, -1)
