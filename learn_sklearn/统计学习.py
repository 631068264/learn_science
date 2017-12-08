#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/4 12:21
@annotation = ''
"""
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
print(data.shape)

digits = datasets.load_digits()
data = digits.images.reshape((digits.images.shape[0], -1))

print(data.shape)

# 拟合数据
# 由scikit学习实现的主要API是 估计量。
# 估计是从数据中学到的任何对象;
# 它可以是从原始数据中提取/过滤有用特征的分类，回归或聚类算法或变换器

# estimator = Estimator(param1=1, param2=2)
# estimator.fit(data)
