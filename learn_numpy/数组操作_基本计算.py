#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/21 16:59
@annotation = '' 
"""
import numpy as np
from numpy import vsplit, dsplit, vstack, hstack, dstack, column_stack, row_stack, hsplit
from numpy.ma import arange, array

"""
切片
"""
a = arange(9)
if False:
    print a
    print a[:7]
    print a[:7:2]

# 三维数组
# 它看做一个两层楼建筑，每层楼有12个房间，并排列成3行4列
b = arange(24).reshape(2, 3, 4)
if False:
    print(b.shape)
    print(b)

"""
改变数组维度
"""
# 展平
if False:
    print(b.ravel())

    print()
    # 多个冒号可以用一个省略号(...)来代替
    print(b[0, ...])  # 首层所有房间

    b.shape = (6, 4)
    print()
    print(b)
    print(b.transpose())  # 转置矩阵

"""
组合
"""
a = arange(9).reshape(3, 3)
if False:
    print(a)
    print()
    b = 2 * a
    print(b)
    print()
    # vstack 垂直
    print(np.vstack((a, b)))
    print()
    # hstack 水平
    print(np.hstack((a, b)))
    # dstack
    print()
    print(dstack((a, b)))

    print(column_stack((a, b)) == hstack((a, b)))
    print(row_stack((a, b)) == vstack((a, b)))

"""
分割
"""
if False:
    print()
    print(a)
    print(vsplit(a, 3))
    print(hsplit(a, 3))

    c = arange(27).reshape(3, 3, 3)
    print(c)
    print(dsplit(c, 3))
"""
遍历
"""
b = arange(4).reshape(2, 2)
if False:
    print b
    for f in b.flat:
        print(f)
"""
NumPy => Python
"""
b = array([1. + 1.j, 3. + 2.j])
if False:
    print(b.tolist())
    # 转换成指定类型
    print(b.astype("complex"))
    print(b.astype(int))

"""
索引数组
"""
a = np.array([1, 2, 3, 4])
# b = array([1, 2, 3, 4, 5, 6])
if False:
    print a[a > 2]
    print np.where(a > 2)

"""
花式索引
"""
a = np.array([
    [1, 2, 3, 4],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
])

if True:
    # 按需取行a
    print a[[3, 2]]
    print a[[-1, -4]]
    print a[[1, 3, 0, 2]][:, [3, 1, 2, 0]]
    # 矩形区域选择器
    print a[np.ix_([1, 3, 0, 2], [3, 1, 2, 0])]
"""
 += 切片 arr.sort(原地排序) 原值运算 np.sort()返回已排序副本
"""
b = a
if False:
    # b = [1 2 3 4]
    # 会新建一个数组给a
    # a = a + array([1, 1, 1, 1])

    # b = [2 3 4 5]
    # 将新值存储在原值所在位置
    # a += array([1, 1, 1, 1])

    # 原值运算 使得切片更有效率
    sli = a[:3]
    sli[0] = 100
    print a
"""
去重 集合操作
"""
a = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 6])
b = np.array([1, 2, 3, 4])
if False:
    print np.unique(a)
    # 交
    print np.intersect1d(a, b)
    # 并
    print np.union1d(a, b)
    # 差
    print np.setdiff1d(a, b)
    # 包含
    print np.in1d(a, b)

"""
基本计算
"""
if False:
    print("算术平均值 =", np.mean([520, 600, 480, 750, 500]))
    print("平均值 =", np.average([520, 600, 480, 750, 500]))
    # ( 80*20% + 90*30% + 95*50% )/(20%+30%+50%）=90.5
    print("加权平均值 =", np.average([80, 90, 95], weights=[.2, .3, .5]))

    # S**2 = 1/n * ((x1 - X)**2 + (x2 - X)**2 + ...) S**2 方差 X 平均分
    # 方差是指各个数据与所有数据算术平均数的离差平方和除以数据个数所得 到的值
    # S 为标准差
    print("方差 =", np.var([80, 90, 95]))
    print("标准差 =", np.std([80, 90, 95]))
    # 期望值 离散型 出现情况*概率
    # 协方差 绝对值越接近1表明线性相关性越好
    print("协方差 = ", np.cov([80, 90, 95], [95, 90, 80]))

    # 取值范围
    h, l = np.loadtxt('data.csv', delimiter=',', usecols=(4, 5), unpack=True)
    print("highest =", np.max([520, 600, 480, 750, 500]))
    print("highest_index =", np.argmax([520, 600, 480, 750, 500]))

    print("lowest =", np.min([520, 600, 480, 750, 500]))
    print("lowest_index =", np.argmin([520, 600, 480, 750, 500]))

    # 计算极差 max(array) - min(array) 计算数组的取值范围
    print("极差 =", np.ptp([520, 600, 480, 750, 500]))
    print("中位数 =", np.median([520, 600, 480, 750, 500]))
    print("排序 =", np.msort([520, 600, 480, 750, 500]))

    # diff函数可以返回一个由相邻数组元素的差 值构成的数组
    print("diff函数 =", np.diff([90, 80, 95]))
    # print(np.log10([10, 100]))

    # 收益率
    c = [90, 80, 95]
    r = np.diff(c) / c[: -1]
    print(r)
    print(np.where(r > 0))
    print(np.take(c, (0, 1)))

    print(np.exp([1, 2, 3]))

    b = np.arange(1, 9)
    print("8! = ", b.prod())
    print("所有阶乘 = ", b.cumprod())


    def my_func(a):
        """Average first and last element of a 1-D array"""
        print(a[0])

        # print(a[-1])
        return (a[0] + a[-1]) * 0.5


    b = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print(np.apply_along_axis(my_func, 0, b))
    # print(np.apply_along_axis(my_func, 1, b))


    print(np.linspace(0, 500, 5))
