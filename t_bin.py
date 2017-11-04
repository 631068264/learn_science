#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/21 23:23
@annotation = ''
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize

a = np.array([
    [0, 0, 2, 5, 0],
    [5, 56, 12, 34, 54],
    [23, 2, 1, 54, 10],
    [2, 0, 1, 27, 0],
])
if False:
    b = binarize(a, threshold=20)
    print b
    print b.sum(axis=1)

if True:
    df = pd.DataFrame(a)
    print df
    mean = df.mean()
    demean_y = df - mean

    resistance_y = demean_y * -1
    support_y = demean_y

