#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/7/3 18:20
@annotation = ''
"""
import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

"""
Note that both misclassify two of the points. 
By default, both models apply an L2 regularization, in the same way that Ridge does for regression.

the strength of the regularization is called C, and higher values of C correspond to less regularization.
C 越大fit越好 越小系数越趋向0更正则化

penalty="l1"
"""
# X, y = mglearn.datasets.make_forge()
# fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#     clf = model.fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
#                                     ax=ax, alpha=.7)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{}".format(clf.__class__.__name__))
#     ax.set_xlabel("Feature 0")
#     ax.set_ylabel("Feature 1")
# axes[0].legend()

"""
Decision boundaries of a linear SVM of different C
"""
# mglearn.plots.plot_linear_svc_regularization()

"""
Linear models for multiclass classification
one-vs.-rest approach


Large dataset solver='sag' in LogisticRegression and Ridge, which can be faster than the default on large datasets

好处:
    Linear models are very fast to train, and also fast to predict.
    linear models is that they make it relatively easy to understand how a prediction is made, 
        using the formulas we saw earlier for regression and classification
    perform well features number larger than samples number 
"""
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.legend(["Class 0", "Class 1", "Class 2"])

linear_clf = LinearSVC().fit(X, y)
linear_clf = LogisticRegression().fit(X, y)
print("Coefficient shape: ", linear_clf.coef_.shape, linear_clf.coef_)
print("Intercept shape: ", linear_clf.intercept_.shape, linear_clf.intercept_)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
print line
print type(line)
for coef, intercept, color in zip(linear_clf.coef_, linear_clf.intercept_,
                                  ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.ylim(-10, 15)
    plt.xlim(-10, 8)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
                'Line class 2'], loc=(1.01, 0.3))

plt.show()
