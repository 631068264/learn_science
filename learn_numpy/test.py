#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/4/18 10:01
@annotation = ''
"""
import numpy as np

# s = [1, 2, 3, 4]
# f = [-1, 2, -3, -1, 2, 6]

# s = [1, 2, 3, 4]
# f = [-1, 2, -3, -1, 2, 3]
#
# slow = np.array(s, dtype=np.float64)
# fast = np.array(f, dtype=np.float64)
#
#
# def fit_series(s1, s2):
#     size = min(len(s1), len(s2))
#     s1, s2 = s1[-size:], s2[-size:]
#     return s1, s2
#
#
# f, s = fit_series(fast, slow)
# print f, s
# a = (f > s)[-2:]
# print a
# b = a[-1]
# print (f > s).any()

# a = np.arange(1, 10)
# b = np.arange(10, 20)
#
# c = np.array([[a], [b]])
# print c
#
# print np.column_stack((a, b))

a = np.array((1, 2, 3))
print a
b = np.array((2, 3, 4))
print np.column_stack((a, b))

test_data = [[], []]
# print len(test_data[0])
stats = {'r': 0, 'w': 0, 'p': {0: 0, 1: 0, -1: 0}, 'a': {0: 0, 1: 0, -1: 0}}

pct_correct = (1.0 * stats['r'] / (stats['r'] + stats['w']))
print pct_correct
for i in range(0, len(test_data[0])):
    print i
