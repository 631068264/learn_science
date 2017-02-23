#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/29 15:50
@annotation = '' 
"""
from datetime import datetime

import numpy as np
import pandas as pd

ts = pd.Series(np.random.randn(10), index=pd.date_range(datetime.now().date(), periods=10))
"""
date_range []
"""
ts = pd.Series(np.random.randn(1000), index=pd.date_range("2017-1-1", periods=1000))

if False:
    # 字符串 datetime 都可以
    print ts
    print ts["20160101":]
    print ts["2017"]
    print ts["2017-01"]

if False:
    """
    截断包括分界点
    """
    # 不要5-1后
    # print ts.truncate(after="2017-5")
    print ts.truncate(before="20170501")

if False:
    dates = pd.DatetimeIndex(["20170103", "20170103", "20170101", "20170102", "20170102", "20170101"])
    ts = pd.Series(np.arange(6), index=dates)
    print ts

if True:
    """
    降采样 考虑区间开闭 边界
    """
    # dates = pd.DatetimeIndex(["20170101", "20170105", "20170107", "20170108", "20170110", "20170112"])
    # ts = pd.Series(np.random.randn(6), index=dates)
    # print ts
    # print ts.resample("d")

    # ts = pd.Series(np.arange(90), index=pd.date_range("2017-1-1", periods=90))
    # # 每月平均
    # print ts.resample("M", kind="period").mean()

    ts = pd.Series(np.arange(12), index=pd.date_range("2017-1-1", periods=12, freq="T"))
    # 每5min sum
    print ts
    print ts.resample("5T").sum()
    print ts.resample("5T").ohlc()
if False:
    """
    升采样 采样频率提高
    """
    frame = pd.DataFrame(np.random.randn(2, 4),
                         index=pd.date_range("20000101", periods=2, freq="W-WED"),
                         columns=list("ABCD"))
    frame2 = frame.resample("D").ffill()
    print frame2["A"]

    frame1 = frame.resample("D").asfreq()
    print frame1

if False:
    """
    偏移
    """
    print pd.date_range("2000-1-1", periods=10, freq="1h30min")
