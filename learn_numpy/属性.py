#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/21 11:19
@annotation = '' 
"""
import numpy as np

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

n = 3

if False:
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    print(c)


def rolling_window(a, window):
    # print a.shape, a.strides
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    # print shape, strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


if True:
    # 数据类型一样
    d = np.arange(7)
    # d = np.array([np.arange(n), np.arange(n), np.arange(n), np.arange(n), np.arange(n), np.arange(n), ])
    # d = np.array(
    #     [[np.arange(n), np.arange(n)],
    #      [np.arange(n), np.arange(n)]]
    # )
    print(d)
    # # 维度
    # print(d.ndim)
    # # 数据类型
    # print(d.dtype)
    # # 不同维度大小
    # print(d.shape)
    # # 数据个数
    # print(d.size)
    # strides指每个轴的下标增加1时数据存储区中的指针所增加的字节数
    # print(d.strides)
    # print (np.lib.stride_tricks.as_strided(d, strides=(8, 8)))
    print(np.max(rolling_window(d, 3), 1))
    print(np.min(rolling_window(d, 3), 1))
    print(np.min(rolling_window(d, 3)))
    print(rolling_window(d, 3))


def n2(n):
    d = np.array([
        np.arange(n),
        np.arange(n)])
    print(d)
    print(d.ndim)
    print(d.dtype)
    print(d.shape)
    print(d.size)


def n3(n):
    d = np.array([
        np.arange(n),
        np.arange(n),
        np.arange(n)])
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
