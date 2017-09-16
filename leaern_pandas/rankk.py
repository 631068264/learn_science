#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/28 17:08
@annotation = ''
"""
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 5],
    'b': [90, 20, 3],
    'c': [5, 1, 15]
})

print df
print df.rank(axis=1)