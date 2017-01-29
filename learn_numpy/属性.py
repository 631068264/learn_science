#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/21 11:19
@annotation = '' 
"""
from numpy.ma import arange, array

"""
ndarray.ndim：给出数组的维数，或数组轴的个数,等于秩.

ndarray.shape：元组中的元素即为NumPy数组每一个维度上的大小。
例如二维数组中，表示数组的“行数”和“列数”。ndarray.shape返回一个元组，这个元组的长度就是维度的数目，即ndim属性。

ndarray.size：数组元素的总个数，等于shape属性中元组元素的乘积。

ndarray.dtype：表示数组中元素类型的对象，可使用标准的Python类型创建或指定dtype。另外也可使用前一篇文章中介绍的NumPy提供的数据类型。

ndarray.itemsize：给出数组中的元素在内存中所占的字节数

你想知道整个数组所占的存储空间，可以用nbytes属性来查看  itemsize和size属性值的乘积

ndarray.data：包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。
"""


def numpysum(n):
    a = arange(n) ** 2
    b = arange(n) ** 3
    c = a + b
    print(c)


def n1(n):
    # 数据类型一样
    d = arange(n)
    print(d)
    print(d.ndim)
    # 数据类型
    print(d.dtype)
    # 数据个数
    print(d.shape)
    print(d.size)


def n2(n):
    d = array([arange(n), arange(n)])
    print(d)
    print(d.ndim)
    print(d.dtype)
    print(d.shape)
    print(d.size)


def n3(n):
    d = array([arange(n), arange(n), arange(n)])
    print(d)
    print(d.ndim)
    print(d.dtype)
    print(d.shape)
    print(d.size)
    print(d[1, 1])


# n1(3)
# print()
# n2(3)
# print()
# n3(3)
a = array([1, 2, 3, 4, 5, 6])
print a[a > 2]
