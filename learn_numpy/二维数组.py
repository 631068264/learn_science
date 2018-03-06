#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/24 21:54
@annotation = '' 
"""
import numpy as np

# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [0, 0, 2, 5, 0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [95, 229, 255, 496, 201],
    [2, 0, 1, 27, 0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])

# Change False to True for each block of code to see what it does

# Accessing elements
if False:
    # print ridership[1, 3] == ridership[1][3]
    # print ridership[1:3, 3:5]
    print ridership[1, :]
    # 按序取行列
    print ridership[[1, 5, 7, 2]][:, [3, 1, 2, 0]]

# Vectorized operations on rows or columns
if False:
    print ridership[0,]
    print ridership[1, ...]
    print ridership[..., 0]
    print ridership[:, 1]

# Vectorized operations on entire arrays
if False:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    print a + b

    a = np.array([1, 2, 3])
    b = np.array([1, 1, 1])
    print a > b


def mean_riders_for_max_station(ridership):
    max_station = ridership[0, :].argmax()
    print max_station
    mean_for_max = ridership[:, max_station].mean()
    overall_mean = ridership.mean()
    return (overall_mean, mean_for_max)


"""
轴 竖0
"""

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print a.shape
if False:
    print a
    print a.sum()
    print a.sum(axis=0)
    print a.sum(axis=1)
if True:
    # 列
    print a[:, 0]
    print a[:, -1]
    # 行
    print a[0, :]
    print a[-1, :]

    print a[-1, 1]

    print a[0, 1]

    print a[0]
    print a[0][1]

if False:
    print np.mean(a[0])
    print a[2]
    print np.square(a)
