#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/24 21:10
@annotation = '' 
"""
from datetime import datetime

import numpy as np
import pandas as pd

"""
与NumPY 不同Pandas 有索引可重复
"""
a = pd.Series([1, "adf", 3.0123, 4], index=["f", "f", "c", "k"])
if False:
    print a
    print a["f"]
    print a["c"]
"""
向量操作 根据index 区分index 和 位置
"""
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([10, 20, 30, 40], index=['c', 'd', 'e', 'f'])
if False:
    print s1 + s2

"""
NaN相关
"""
if False:
    s = s1 + s2
    print s
    # 结果补0
    print s.fillna(0)
    # 去掉含nan的行
    print s.dropna()
    # 操作前缺失的地方补0
    print s1.add(s2, fill_value=0)

if False:
    s1 = pd.Series([1, 2, 3, None], index=['a', 'b', 'c', 'd'])
    print s1.isnull()
    print s1[s1.notnull()] == s1.dropna()

"""
删除索引
"""
if False:
    c = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
    print c
    d = c.drop("d")
    print d
    a = d.append(pd.Series([5]), ignore_index=True)
    print a
    a = d.append(pd.Series([5], index=["d"]))
    print a

    c["a"] = 8
    print c
    print c["a"]

if False:
    """
    增删改
    """
    df = pd.DataFrame(
        [
            [9, 6, 2, 3],
            [9, "sdf", 2, 3],
            [5, datetime.now(), np.NaN, 10],
            [.89, "", 3, None],
        ],
        index=list("fuck"),
        columns=list("ABCD"),
    )
    a = df.append(
        pd.Series(["fs", "df", 12, 2312], name="哈哈", index=list("ABCD"))
    )
    print a
    d = a.drop("哈哈")
    print d

    d = df.drop("A", axis=1)
    print d

    df.loc["f"] = pd.Series(["fs", "df", 12, 2312], index=list("ABCD"))
    print df
    df["B"] = pd.Series(["fs", "df", 12, 2312], index=list("fuck"))
    print df

if False:
    """
    切片
    """
    s1 = np.array([1, 2, 3, 4])
    print s1[1:3]
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    print s1[1:3]
    """
    闭区间
    """
    print s1["a":"c"]
    # 取 a c
    print s1[["a", "c"]]

if True:
    """
    选取
    """
    df = pd.DataFrame(
        np.arange(16).reshape(4, 4),
        index=['f', 'u', 'c', 'k', ],
        columns=['A', 'B', 'C', 'D', ]
    )
    # 按列取
    print df
    print df[["A", "C"]]
    print df["A"] == df.A

    print '行 index'
    # print df[0]
    print df[0:2]
    print df.iloc[2:]
    print df.iloc[2] == df.loc["c"]
    '''
    df.loc[行标签,列标签]
    df.loc['a':'b'] #选取 ab 两行数据
    df.loc[:,'open'] #选取 open 列的数据
    '''
    print '区域索引'
    print df.ix["k", "A"]
    print df.ix[["f", "k"],]
    print df.ix["f":"k", "A"]

    print "逻辑"
    print df[(df.A > 3) & (df.C > 0)]
    print df[df.A.isin([4, 8])]
    # print df[df.loc["c"] > 3]
"""
drop index
"""
