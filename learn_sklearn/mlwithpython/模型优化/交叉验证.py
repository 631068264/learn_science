#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/5 17:24
@annotation = ''
"""
from sklearn.datasets import load_iris, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

"""
交叉验证
k-fold cross-validation
the data is first partitioned into K parts of (approximately) equal size called folds

k=5
    1 for test,2-5 for train
    2 for test 1345 for train
    ....
    In the end, we have collected five accuracy values.
"""

iris = load_iris()
logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("Cross-validation scores: \n{}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))

"""
When using train_test_split, we usually use 75% of the data for training and 25% of the data for evaluation. 
When using five-fold cross-validation, in each iteration we can use four-fifths of the data (80%) to fit the model.
"""

"""
classifier
    分层stratified k-fold cross-validation 对付dataset有分类极端确保train
For regression 
    scikit-learn uses the standard k-fold cross-validation by defaul
"""
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
print("Cross-validation scores:\n{}".format(cross_val_score(logreg, iris.data, iris.target, cv=kfold)))
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
print("Cross-validation scores:\n{}".format(cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

StratifiedShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("Cross-validation scores:\n{}".format(scores))

"""
groups specifies which group (think patient) the point belongs to

KFold, StratifiedKFold, and GroupKFold
"""
from sklearn.model_selection import GroupKFold

# create synthetic dataset
X, y = make_blobs(n_samples=12, random_state=0)
# assume the first three samples belong to the same group, # then the next four, etc.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=5))
print("Cross-validation scores:\n{}".format(scores))
