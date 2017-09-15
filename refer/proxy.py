#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/13 18:02
@annotation = ''
"""
import sys
import weakref


class Man:
    def __init__(self, name):
        self.name = name

    def test(self):
        print "this is a test!"


def callback(self):
    print "callback"


o = Man('Jim')
p = weakref.proxy(o, callback)
# r = weakref.ref(o)
# d = r()
p.test()
print sys.getrefcount(o)
o = None
p.test()
