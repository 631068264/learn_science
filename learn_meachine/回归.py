#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/5 14:12
@annotation = ''
"""
from __future__ import print_function

import numpy as np
from sklearn import datasets

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

if False:
    print(diabetes_X_test.shape)

    print(diabetes_X_test.dtype)

    print(diabetes_y_test.shape)

    print(diabetes_X_test.dtype)

    from sklearn import linear_model

    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    # 系数
    print(regr.coef_)

    # The mean square error
    print(np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

    # Explained variance score: 1 is perfect prediction
    # and 0 means that there is no linear relationship
    # between X and y.
    # 可解释变异（英语：explained variation）在统计学中是指给定数据中的变异能被数学模型所解释的部分。通常会用方差来量化变异，故又称为可解释方差（explained variance）。
    # 除可解解变异外，总变异的剩余部分被称为未解释变异（unexplained variation）或残差（residual）
    print(regr.score(diabetes_X_test, diabetes_y_test))


    # regr.fit(np.array([1, 2, 4], dtype="float64"), np.array([8], dtype="float64"))
    # print regr.coef_
    # print regr.predict(np.array([1, 2, 4], dtype="float64"))
    # print regr.score(np.array([[1, 2, 4]], dtype="float64"), np.array([8], dtype="float64"))

if True:
    # 收缩
    """每个维度数据少"""
    # If there are few data points per dimension, noise in the observations induces high variance
    X = np.c_[.5, 1].T
    y = [.5, 1]
    test = np.c_[0, 2].T

    from sklearn import linear_model
    import matplotlib.pyplot as plt

    regr = linear_model.LinearRegression()
    # 正则化
    # 高维统计学习一种解决方案是收缩回归系数为零：观察中的任何两个随机选择的组可能是不相关的。这被称为Ridge 回归
    # 偏差/方差折衷的一个例子：脊 alpha参数越大，偏差越高，方差越小
    # regr = linear_model.Ridge(alpha=.1, normalize=True)

    # 缩小对系数影响 选择系数
    regr = linear_model.Lasso(alpha=.001, normalize=True)
    plt.figure()
    np.random.seed(0)
    for _ in range(6):
        this_X = .1 * np.random.normal(size=(2, 1)) + X
        regr.fit(this_X, y)
        plt.plot(test, regr.predict(test))
        plt.scatter(this_X, y, s=3)
    # plt.show()
    alphas = np.logspace(-4, -1, 6)

    print([regr.set_params(alpha=alpha
                           ).fit(diabetes_X_train, diabetes_y_train,
                                 ).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])
    plt.show()
# 高效离散
# Different algorithms can be used to solve the same mathematical problem. For instance the Lasso object in scikit-learn
# solves the lasso regression problem using a coordinate decent method, that is efficient on large datasets.
# However, scikit-learn also provides the LassoLars object using the LARS algorthm, which is very efficient for problems
#     in which the weight vector estimated is very sparse (i.e. problems with very few observations).
