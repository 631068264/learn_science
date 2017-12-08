#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/7/15 10:10
@annotation = ''
"""
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

X, y = make_blobs(centers=4, random_state=8)
y = y % 2


# print X
# print
# print y

def a():
    linear_svm = LinearSVC().fit(X, y)
    mglearn.plots.plot_2d_separator(linear_svm, X)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


"""
核函数 计算合适的多项式和合适的degree

高斯核Gaussian kernel 合适无限维度
Gaussian kernel is that it considers all possible polynomials of all degrees, but the importance of the features decreases for higher degrees

support vectors 子集与训练集决策边界

分类决策
A classification decision is made based on the distances to the support vector,
and the importance of the support vectors that was learned during training (stored in the dual_coef_ attribute of SVC)
"""


def b():
    X, y = mglearn.tools.make_handcrafted_dataset()
    """
    参数调优
        C & gamma
        gamma 控制函数的宽度 点之间的距离
        C regularization 参数 与线性模型相近 It limits the importance of each point (or more precisely, their dual_coef_)
        
        A small gamma means a large radius for the Gaussian kernel, which means that many points are considered close by.
        smooth decision boundaries
        a small C means a very restricted model
    """
    svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)

    # plot support vectors
    sv = svm.support_vectors_
    # class labels of support vectors are given by the sign of the dual coefficients
    sv_labels = svm.dual_coef_.ravel() > 0
    print sv
    print sv_labels
    print
    print svm.dual_coef_
    print
    print svm.dual_coef_.ravel()
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


def c():
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, C in zip(axes, [-1, 0, 3]):
        for a, gamma in zip(ax, range(-1, 2)):
            mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
    axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
                      ncol=4, loc=(.9, 1.2))
    plt.show()


def d():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
    svc = SVC()
    svc.fit(X_train, y_train)
    """
    MinMaxScaler preprocessing can get a better result
    """
    print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
    print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

    plt.plot(X_train.min(axis=0), 'o', label="min")
    plt.plot(X_train.max(axis=0), '^', label="max")
    plt.legend(loc=4)
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    plt.yscale("log")
    plt.show()


d()

"""
缺点
    不同维度都有良好表现 但是数据量大>100,000 消耗时间和内存
    需要谨慎预处理数据与调优参数

"""