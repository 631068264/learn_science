#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/7/15 22:02
@annotation = ''
"""
import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)


def a():
    mlp = MLPClassifier(solver='lbfgs', activation='tanh',
                        random_state=0, hidden_layer_sizes=[10, 10])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


def b():
    """
    控制点
        hidden layers num, unit num in hidden layer,regularization (alpha)
    inputs need scale the data

    increasing max_iter only increased the training set performance,
    not the generalization performance

    increase alpha decrease model’s complexity and stronger regularization of the weights
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for axx, n_hidden_nodes in zip(axes, [10, 100]):
        for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
            mlp = MLPClassifier(solver='lbfgs', random_state=0,
                                hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                                alpha=alpha)
            mlp.fit(X_train, y_train)

            mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
            mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
            ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(
                n_hidden_nodes, n_hidden_nodes, alpha))
    plt.show()


def c():
    """
    优点
        处理大数据 建立极其复杂的模式
        消耗时间 数据 优化参数 结果better > 其他算法

    可以轻易计算模型复杂度

    先overfit确保网络能够学习 再缩小网络或者increase alpha 或者 regularization improve generalization performance

    选择solver
        adam
            Works well in most situations but is quite sensitive to the scaling of the data
        (so it is important to always scale your data to 0 mean and unit variance)

        l-bfgs
            更加健壮性 take a time


    """
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
    mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
    mlp.fit(X_train, y_train)

    print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
    print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
    print("Accuracy weight".format(mlp.coefs_[0]))


def d():
    X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
    # we rename the classes "blue" and "red" for illustration purposes
    y_named = np.array(["blue", "red"])[y]

    # we can call train_test_split with arbitrarily many arrays;
    # all will be split in a consistent manner
    X_train, X_test, y_train, y_test, y_train, y_test = \
        train_test_split(X, y_named, y, random_state=0)
    # build the gradient boosting model
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train)

    # print("X_test.shape: {}".format(X_test.shape))
    # print("Decision function shape: {}".format(
    #     gbrt.decision_function(X_test).shape))

    # show the first few entries of decision_function
    # print("Thresholded decision function:\n{}".format(gbrt.decision_function(X_test) > 0))
    greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
    print("分类类型 :\n{}".format(gbrt.classes_))
    print("Shape of probabilities: \n{}".format(gbrt.predict_proba(X_test)))
    print("decision function: \n{}".format(gbrt.decision_function(X_test)))
    print("Predictions:\n{}".format(gbrt.predict(X_test)))
    # print("pred is equal to predictions:{}".format(np.all(gbrt.classes_[greater_zero] == gbrt.predict(X_test))))
    """
    决策边界
    decision_function
        it returns one floating-point number for each sample
    预测可能性
    predict_proba
        a probability for each class, and 
        is often more easily understood than the output of decision_function
        列举说有分类的可能性 sum=1
        大于50%作为predict结果
    
    overfit and 复杂 预测的准确性更高
    
    """
    # plot_dicision(X, X_test, X_train, gbrt, y_test, y_train)

    plot_probla(X, X_test, X_train, gbrt, y_test, y_train)


def plot_probla(X, X_test, X_train, gbrt, y_test, y_train):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mglearn.tools.plot_2d_separator(
        gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
    scores_image = mglearn.tools.plot_2d_scores(
        gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')
    for ax in axes:
        # plot training and test points
        mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                                 markers='^', ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                                 markers='o', ax=ax)
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        cbar = plt.colorbar(scores_image, ax=axes.tolist())
        axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
                        "Train class 1"], ncol=4, loc=(.1, 1.1))
    plt.show()


def plot_dicision(X, X_test, X_train, gbrt, y_test, y_train):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Decision boundary
    mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
                                    fill=True, cm=mglearn.cm2)
    # decision function
    scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                                alpha=.4, cm=mglearn.ReBl)
    for ax in axes:
        # plot training and test points
        mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                                 markers='^', ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                                 markers='o', ax=ax)
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        cbar = plt.colorbar(scores_image, ax=axes.tolist())
        axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
                        "Train class 1"], ncol=4, loc=(.1, 1.1))
    plt.show()


d()
