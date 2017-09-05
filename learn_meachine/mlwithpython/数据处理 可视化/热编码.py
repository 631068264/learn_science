#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/1 16:34
@annotation = ''
"""
import pandas as pd

if False:
    """
    one-hot encoding
    """
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': ['b', 'a', 'c'],
        'C': [1, 2, 3],
    })

    """
    check if a column actually contains meaningful categorical data.
    checking string-encoded categorical data
    """
    print df
    print df['A'].value_counts()

    """
    value_counts function of a pandas Series (the type of a single column in a DataFrame
    
    """
    dummy = pd.get_dummies(df)
    print dummy
    print dummy.columns
    print dummy.values

if False:
    """
    数字编码
    """
    demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                            'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
    print demo_df
    print pd.get_dummies(demo_df)

    demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
    print pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])
