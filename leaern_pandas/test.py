#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/24 20:55
@annotation = '' 
"""
import pandas as pd

a = pd.Series([1, 2, 3, 4])

# Accessing elements and slicing
if False:
    for country_life_expectancy in a:
        print 'Examining life expectancy {}'.format(country_life_expectancy)  # Pandas functions
# if False:
#     print a[0]
#     print a[3:6]
#     print a.mean()
#     print a.std()
#     print a.max()
#     print a.sum()

if False:
    a = pd.Series([1, 2, 3, 4])
    c = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
    b = pd.Series([1, 2, 1, 2])
    print a.values
    print a.index
    print a + b
    print a * 2
    print a >= 3
    print a[a >= 3]

if False:
    d = pd.Series({
        "a": 1,
        "b": 2,
    })
    print d


# def variable_correlation(variable1, variable2):
#     both_above = (variable1 > variable1.mean()) & (variable2 > variable2.mean())
#     both_below = (variable1 < variable1.mean()) & (variable2 < variable2.mean())
#
#     num_same_direction = (both_above | both_below).sum()
#     num_different_direction = len(variable1) - num_same_direction
#
#     return (num_same_direction, num_different_direction)


# print variable_correlation(life_expectancy, gdp)

"""

"""

entries_and_exits = pd.DataFrame({
    'ENTRIESn': [3144312, 3144335, 3144353, 3144424, 3144594,
                 3144808, 3144895, 3144905, 3144941, 3145094],
    'EXITSn': [1088151, 1088159, 1088177, 1088231, 1088275,
               1088317, 1088328, 1088331, 1088420, 1088753]
})


def get_hourly_entries_and_exits(entries_and_exits):
    print (entries_and_exits - entries_and_exits.shift(1)).dropna()

# get_hourly_entries_and_exits(entries_and_exits)
