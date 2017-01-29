#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/25 10:19
@annotation = '' 
"""
from datetime import datetime

import numpy as np
import pandas as pd

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
"""
DataFrame
"""
if False:
    df.to_csv("my.csv", header=None)
    print pd.read_csv("my.csv", sep=",", names=list("abcd")).ix["c", "b"]
    html_content = df.to_html()

    for d in pd.read_html(html_content):
        print d
"""
csv
"""
if False:
    sw_df = pd.read_csv('nyc-subway-weather.csv')
    print sw_df.head(5)

"""
基本读写 二进制
"""
if False:
    a = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 6])
    if False:
        np.save("fd", a)
        print np.load("fd.npy")
    if False:
        # 压缩
        np.savez("fd.npz", a=a)
        print np.load("fd.npz")["a"]

"""
文本文件
"""
# 创建单位矩阵
if False:
    i2 = np.eye(3)
    print(i2)
    np.savetxt("eye.txt", i2, fmt="%s")

    # 分隔符为, usecols的参数为一个元组，以获取第7字段至第8字段的数据
    # unpack参数设置为True，意思是分拆存储不同列的数据，即分别将收 盘价和成交量的数组赋值给变量c和v
    c, v = np.loadtxt('data.csv', delimiter=',', usecols=(6, 7), unpack=True)
    print((c, v))

if False:
    d = {
        "account_key": [1, 2, 3, 4],
        "date": [1, 2, 3, 4],
        "time": [None, None, 3, None],
    }

    print pd.DataFrame(d).to_json()
