#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/27 11:33
@annotation = '' 
"""
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 1, 2])

if False:
    print a > a.mean()
    print (a > a.mean()).all()
    print (a > a.mean()).any()
    print a > a.mean()
    print ~(a > a.mean())
    both_above = (a > a.mean()) & (b > b.mean())
    both_below = (a < a.mean()) & (b < b.mean())
    print both_above
    print both_below
if False:
    print a[a > 2]
    # false 选b
    print np.where(np.array([False, False, True, True]), a, b)
    print np.where(a > 2, 100, 0)
