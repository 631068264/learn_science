#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/28 16:51
@annotation = '' 
"""
import pandas as pd

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 36, 60, 100]
if False:
    """
    ages 落在 bins 区间
    """
    cats = pd.cut(ages, bins, right=False)
    print cats
    cats = pd.cut(ages, bins)
    print cats
    # 区间index
    print cats.codes
    # print cats.levels
    print pd.value_counts(cats)
