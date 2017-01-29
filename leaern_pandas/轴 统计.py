#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/27 18:53
@annotation = '' 
"""
import pandas as pd

df = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})

# Pandas axis
if False:
    print df
    # 按列
    print df.sum()
    # 行
    print df.sum(axis=1)
    # 全
    print df.values.sum()
if False:
    s1 = pd.Series([1, 4, 3, 4, 3], index=list("abcke"))
    print s1.unique()
    print s1.value_counts(sort=False)
    print s1.value_counts()
    print s1.isin([6, 3])

"""
轴向旋转 DataFrame Series
"""
s1 = pd.Series([0, 1], index=list("ab"))
s2 = pd.Series([2, 3, 4], index=list("cde"))

if False:
    # stack 会过滤NaN
    s = df.stack()
    print s
    print s[0, "A"]
    print s.unstack()
    # d = s1.unstack()
    # print d

if True:
    s = pd.concat([s1, s2], keys=list("AB"))
    print s
    print s.unstack().stack()
