#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/16 21:25
@annotation = ''
"""
import pandas as pd

df = pd.DataFrame({'id': [1, 2, 3, 4],
                   'treatment': ['A', 'A', 'B', 'B'],
                   'gender': ['F', 'M', 'F', 'M'],
                   'response': [5, 3, 8, 9],
                   })
# print df

multi_index = df.set_index(['treatment', 'gender'])
print multi_index

# print multi_index.index.names

a = multi_index.unstack(level="gender")
b = multi_index.unstack(level=1)
# print a == b
print a
# print a.index.names, a.columns
