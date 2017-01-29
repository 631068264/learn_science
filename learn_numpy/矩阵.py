#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/27 11:33
@annotation = ''
"""
import numpy as np

A = np.mat('1 2 3; 4 5 6; 7 8 9')
if False:
    print("Creation from string", A)
    print("transpose A", A.T)
    print("Inverse A", A.I)
    print("Check Inverse", A * A.I)

    print("Creation from array", np.mat(np.arange(9).reshape(3, 3)))
    # print("Creation from array", np.arange(9).reshape(3, 3).T)
    print(np.mat("-2 1;4 -3").I)
"""
矩阵相乘 行列积相加
"""
a = np.array([
    [2, 1],
    [4, 3]
])
b = np.array([
    [1, 2],
    [1, 0]
])

if False:
    print a.T
    print b.T
    print a.dot(b)

nwalks = 5000
nsteps = 1000
# draws = np.random.randint(0, 2, nsteps)
draws = np.random.randint(0, 2, (nwalks, nsteps))
print draws
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
print walk
