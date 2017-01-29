#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/25 15:18
@annotation = '' 
"""
import pandas as pd

subway_df = pd.DataFrame({
    "account_key": [1, 2, 3, 4],
    "date": [1, 2, 3, 4],
    "time": [None, None, 3, None],
})
weather_df = pd.DataFrame({
    "account_key": [1, 2, 4, 1],
    "date_n": [1, 2, 3, 4],
    "project": [None, None, 3, None],
})

if False:
    print subway_df
    print weather_df

if False:
    """
    merge 和 SQL join 差不多
    on 名称不一致用left_on, right
    """
    print subway_df.merge(weather_df, left_on=["account_key", "date"], right_on=["account_key", "date_n"], how="inner")

    print subway_df.merge(weather_df, on="account_key", how="left")

    # print subway_df.merge(weather_df, on="account_key", how="left", right_index=True)

s1 = pd.Series([0, 1], index=list("ab"))
s2 = pd.Series([2, 3, 4], index=list("cde"))
s3 = pd.Series([5, 6], index=list("fg"))

if False:
    s = pd.concat([s1, s2, s3])
    print s
    print s3[1]
