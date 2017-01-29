#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/27 11:08
@annotation = '' 
"""
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
if True:
    print a
    if (a.T == a.transpose(1, 0)).all():
        print "ok"

ar = np.arange(16).reshape(2, 2, 4)

if False:
    print ar
    print ar.transpose(1, 0, 2)
