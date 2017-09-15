#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/13 17:54
@annotation = ''
"""

import sys
import weakref


class Man(object):
    def __init__(self, name):
        self.name = name


o = Man('Jim')
print sys.getrefcount(o)
r = weakref.ref(o)  # 创建一个弱引用
print sys.getrefcount(o)  # 引用计数并没有改变

# 弱引用所指向的对象信息
o2 = r()  # 获取弱引用所指向的对象
print o is o2
print sys.getrefcount(o)

o = None
print r
print sys.getrefcount(o)
o2 = None
print r  # 当对象引用计数为零时，弱引用失效。
