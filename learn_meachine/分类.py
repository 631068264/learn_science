#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/4 17:32
@annotation = ''
"""
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# 随机排列，随机分割数据
# np.random.seed(0)
indices = np.random.permutation(len(iris_X))
# train_test_split()
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

print iris_X_test.shape
print iris_X_test.dtype
print iris_y_test.shape
print iris_y_test.dtype

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
"""
x 推测 y

fit 拟合x,y的train
推测 x_test 对应的y_test

估计器有效，您需要相邻点之间的距离小于某个值d
"""
knn.fit(iris_X_train, iris_y_train)
print(knn.predict(iris_X_test))
print(knn.score(iris_X, iris_y))
print(knn.score(iris_X_test, iris_y_test))


# logistic = linear_model.LogisticRegression(C=1e5)
# logistic.fit(iris_X_train, iris_y_train)
