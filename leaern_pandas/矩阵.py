#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/21 11:38
@annotation = ''
"""
import numpy as np
import pandas as pd

df = pd.DataFrame(
    np.arange(16).reshape(4, 4),
    index=list('fuck'),
    columns=list('ABCD')
)

# print df
matrix = df.as_matrix()
print matrix
X = matrix[:, 1:]
y = matrix[:, 0]
print X
print
print y
