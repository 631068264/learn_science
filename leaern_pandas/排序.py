#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/27 18:33
@annotation = '' 
"""
import pandas as pd

s1 = pd.Series([1, 2, 3, 4], index=['b', 'a', 'c', 'd'])

if False:
    # 降序
    print s1.sort_values(ascending=False)
    print s1.sort_index()

df = pd.DataFrame(
    [
        [9, 6, 2, 3],
        [10, 8, 2, 3],
        [5, 1, 3, 10],
        [5, 1, 3, 10],
    ],
    index=list("fuck"),
    columns=list("ABCD"),
)
if True:
    print df
    # 根据列排序决定决定行位置
    print df.sort_values(by=["A", "B"])
    print df.sort_index()
    print(df['A'].argmax())
