#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/15 15:14
@annotation = ''
"""


class Obj(object):
    __slots__ = ('i', 'l')

    def __init__(self, i):
        self.i = i
        self.l = []


all = {}
for i in range(1000000):
    all[i] = Obj(i)
