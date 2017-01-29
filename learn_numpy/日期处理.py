#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/21 22:52
@annotation = '' 
"""
from datetime import datetime

import numpy as np


def datestr2num(s):
    return datetime.strptime(s.decode("utf-8"), "%d-%m-%Y").date().weekday()


dates, close = np.loadtxt('data.csv', delimiter=',', usecols=(1, 6),
                          converters={1: datestr2num}, unpack=True)

# ValueError: could not convert string to float: b'28-01-2011'
# 日期尝试str转浮点 报错
print(dates)

averages = np.zeros(5)
averages[1] = 1
# print(averages)
