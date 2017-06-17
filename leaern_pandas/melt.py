#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/16 14:59
@annotation = ''
"""
import pandas as pd

df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})
print df
"""列属性重复"""
df_melt = pd.melt(frame=df, id_vars=['A'], value_vars=['B', 'C'], var_name='type', value_name='value')
print df_melt
"""行属性重复"""
df_pivot = df_melt.pivot(index="A", columns="type", values="value")
print df_pivot

print df_pivot.columns
print df_pivot.info()
