#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/24 22:29
@annotation = '' 
"""
import numpy as np
import pandas as pd

"""
自定义函数 有点像map
"""

s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])


def add3(x):
    return x + 3


if False:
    print s1.apply(add3)
"""
applymap()

pandas series 即 DataFrame 新列
apply() 针对每一列
"""

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [10, 20, 30],
    'c': [5, 10, 15]
})


def add_one(x):
    return x + 1


def get_second_max(column):
    return column.sort_values(ascending=False).iloc[1]


if False:
    print df.applymap(add_one)

    print df.apply(np.max)
    print df.apply(get_second_max)

grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio',
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)


def std_cloumn():
    print grades_df.mean(axis=0) == grades_df.mean(axis='index')
    return (grades_df - grades_df.mean(axis='index')) / grades_df.std()


def std_row():
    print grades_df.mean(axis=1) == grades_df.mean(axis='columns')
    return (grades_df.sub(grades_df.mean(axis=1), axis=0)).div(grades_df.std(axis=1), axis=0)


"""
DataFrame Series 按照列相加 没有对应的列名columns就NaN

Series 的index 对应 DataFrame的列名
"""

if False:
    s = pd.Series([1, 2, 3, 4], index=['b', 'a', 'c', 'd'])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })

    print df
    print ''  # Create a blank line between outputs
    print df + s

if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })

    print df
    print ''  # Create a blank line between outputs
    print df + s
