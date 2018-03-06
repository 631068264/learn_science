#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/28 17:08
@annotation = ''
"""
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'a': [5, 1, 2,5],
    'b': [90, 20, 3,3],
    'c': [5, 1, 15,0]
})
d = np.linspace(0, 1, df.shape[0])
# print df
# print d
# weights_cnt =3
# weights = weights_cnt * [1. / weights_cnt, ]
# print weights
# print (df['a'].rank().values -1).astype(int) * weights
# print d[(df['a'].rank().values -1).astype(int)]
df.sort_values('a',inplace=True)
print df['a'].value_counts()
# print df['a'].index[-1]