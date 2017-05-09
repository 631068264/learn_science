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

s = [1, 2, 3, 4]
f = [-1, 2, -3, -1, 2, 3]

slow = np.array(s, dtype=np.float64)
fast = np.array(f, dtype=np.float64)


def fit_series(s1, s2):
    size = min(len(s1), len(s2))
    s1, s2 = s1[-size:], s2[-size:]
    return s1, s2


f, s = fit_series(fast, slow)
print f, s
a = (f > s)[-2:]
print a
b = a[-1]
print (f > s).any()
