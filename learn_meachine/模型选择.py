#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/5 16:08
@annotation = ''
"""
from __future__ import print_function

import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import KFold

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')

if False:
    # score可以判断新数据的拟合质量（或预测）的方法。越大越好
    print(svc)
    print(svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:]))

if False:
    """交叉验证"""
    # 为了更好地测量预测精度（我们可以将其用作模型的拟合优势），我们可以连续地将数据分割成用于训练和测试的折叠
    X_folds = np.array_split(X_digits, 3)
    y_folds = np.array_split(y_digits, 3)
    scores = list()
    for k in range(3):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)
        scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
    print(scores)

if True:
    """Cross-validation generators"""

    X = ["a", "a", "b", "c", "c", "c"]
    k_fold = KFold(n_splits=5)
    for train_indices, test_indices in k_fold.split(X):
        print('Train: %s | test: %s' % (train_indices, test_indices))
    """test score"""
    print(cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1))

if False:
    """grid"""

    Cs = np.logspace(-6, -1, 10)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),
                       n_jobs=-1)
    clf.fit(X_digits[:1000], y_digits[:1000])

    print(clf.best_score_)

    print(clf.best_estimator_.C)
    print(clf.best_estimator_)

    # Prediction performance on test set is not as good as on train set
    print(clf.score(X_digits[1000:], y_digits[1000:]))

    print(cross_val_score(clf, X_digits, y_digits))