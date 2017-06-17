#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/15 16:22
@annotation = ''
"""
import matplotlib.pyplot as plt

year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# line plot
plt.plot(year, pop)
# 散点图
plt.scatter(year, pop)


plt.hist()
# plt.show()


print help(plt.hist)