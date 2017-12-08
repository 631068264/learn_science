#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/6 18:14
@annotation = ''
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target, random_state=0)
# print("Size of training set: {} size of test set: {}".format(X_train.shape[0], X_test.shape[0]))
#
# best_score = 0
# for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
#     for C in [0.001, 0.01, 0.1, 1, 10, 100]:
#         # for each combination of parameters, train an SVC
#         svm = SVC(gamma=gamma, C=C)
#         svm.fit(X_train, y_train)
#         # evaluate the SVC on the test set
#         score = svm.score(X_test, y_test)
#         # if we got a better score, store the score and parameters
#         if score > best_score:
#             best_score = score
#             best_parameters = {'C': C, 'gamma': gamma}
# print("Best score: {:.2f}".format(best_score))
# print("Best parameters: {}".format(best_parameters))

"""
Because we used the test data to adjust the parameters, 
we can no longer use it to assess how good the model is

将dataset 分三份 train test valid

it is important to keep a separate test set, which is only used for the final evaluation

It is good practice to do all exploratory analysis and model selection 
using the combination of a training and a validation set, 
and reserve the test set for a final evaluation
"""

# split data into train+validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1)
print("Size of training set: {} size of validation set: {} size of test set:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
best_score = 0

if False:
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            # for each combination of parameters, train an SVC
            svm = SVC(gamma=gamma, C=C)
            svm.fit(X_train, y_train)
            # evaluate the SVC on the test set
            score = svm.score(X_valid, y_valid)
            # if we got a better score, store the score and parameters
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}

    # rebuild a model on the combined training and validation set, # and evaluate it on the test set
    svm = SVC(**best_parameters)
    svm.fit(X_trainval, y_trainval)
    test_score = svm.score(X_test, y_test)
    print("Best score on validation set: {:.2f}".format(best_score))
    print("Best parameters: ", best_parameters)
    print("Test set score with best parameters: {:.2f}".format(test_score))

"""
grid search by cv
"""
if False:
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            # for each combination of parameters,
            # train an SVC
            svm = SVC(gamma=gamma, C=C)
            # perform cross-validation
            scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)  # compute mean cross-validation accuracy
            score = np.mean(scores)
            # if we got a better score, store the score and parameters
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}
    # rebuild a model on the combined training and validation set
    svm = SVC(**best_parameters)
    svm.fit(X_trainval, y_trainval)
    test_score = svm.score(X_test, y_test)
    print("Best score on validation set: {:.2f}".format(best_score))
    print("Best parameters: ", best_parameters)
    print("Test set score with best parameters: {:.2f}".format(test_score))

if True:
    # param_grid = {
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #     'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    # }
    param_grid = [
        {
            'kernel': ['rbf'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        {
            'kernel': ['linear'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
        }
    ]
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    """
    GridSearchCV will use cross-validation in place of the split into a training and validation set that we used before.
    However, we still need to split the data into a training and a test set, to avoid overfitting the parameters
     
    we can call the standard methods fit, predict, and score on it 
    However, when we call fit, it will run cross-validation for each combination of parameters in param_grid
    
    fit运行过程method 和grid_search by cv 差不多
    """
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0)
    grid_search.fit(X_train, y_train)
    print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    print("Best estimator:\n{}".format(grid_search.best_estimator_))
    """a dictionary storing all aspects of the search"""
    results = pd.DataFrame(grid_search.cv_results_)
    print results.head()
