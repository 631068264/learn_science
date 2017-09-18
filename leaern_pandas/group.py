#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/25 14:19
@annotation = '' 
"""
import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3
},
    # index=['a', 'b', 'c', 'd', 'e', 'f', 'g']
)

"""
列 group_by
"""
if False:
    print example_df
    grouped_data = example_df.groupby(['even', 'above_three'])
    print grouped_data.groups

if False:
    print example_df
    grouped_data = example_df.groupby('even')
    print grouped_data.groups
    print grouped_data.sum()
    print
    print grouped_data.sum()["value"]
    print
    print grouped_data["value"].sum()

df = pd.DataFrame({
    "key1": list("aabba"),
    "key2": list("ABABA"),
    "data1": [1] * 5,
    "data2": [2] * 5,
})
if True:
    """
    groupby 选取
    """
    print df
    # 单列 groupby
    print df["data1"].groupby(df["key1"])
    print df["data1"].groupby(df["key1"]).groups
    print df["data1"].groupby(df["key1"]).mean()
    # 全groupby
    print df.groupby("key1")["data2"].mean()

if False:
    """
    groupby 迭代
    """
    # print df
    for (k1, k2), group in df.groupby(["key1", "key2"]):
        print (k1, k2)
        print group
    # DataFrame dict
    pi = dict(list(df.groupby(["key1"])))
    print pi["a"]

if False:
    """
    聚合 agg
    """
    print df.groupby(["key1", "key2"])["data1"].agg("mean")
    print df.groupby(["key1", "key2"])["data1"].agg(["mean", "std"])
    print df.groupby(["key1", "key2"]).agg(["mean", "std"])
    print df.groupby(["key1", "key2"]).agg([("data1", np.mean), ("data2", np.std)])
    print df.groupby(["key1", "key2"]).agg([("data1", np.mean), ("data2", np.std)])["data1"]

if True:
    print df.groupby("key1").mean().add_prefix("mean_")
