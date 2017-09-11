#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/11 10:20
@annotation = ''
"""

from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

"""
The most common use case of the Pipeline class is in chaining preprocessing steps
"""
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
"""
按照pipe顺序来 fit score 
"""
if False:
    pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
    pipe.fit(X_train, y_train)
    print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))

if False:
    pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
    # 与上面pipe函数有关
    param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
    print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
    print("Best parameters: {}".format(grid.best_params_))

"""
pipeline 
fit 
    前面step都执行fit_transform or fit transform 最后一步执行 fit
predict/score
    前面step X 执行transform 最后一步执行predict/score
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

if False:
    # standard syntax
    pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
    # abbreviated syntax
    pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
    print("Pipeline steps:\n{}".format(pipe_short.steps))

    pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
    print("Pipeline steps:\n{}".format(pipe.steps))
    # fit the pipeline defined before to the cancer dataset
    pipe.fit(cancer.data)
    # extract the first two principal components from the "pca" step
    components = pipe.named_steps["pca"].components_
    print("components.shape: {}".format(components.shape))

"""
    pipe and grid-search
"""
if False:
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=4)
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best parameters: {}".format(grid.best_params_))
    print("Best estimator:\n{}".format(grid.best_estimator_))
    print("Logistic regression coefficients:\n{}".format(grid.best_estimator_.named_steps["logisticregression"].coef_))

if False:
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                        random_state=0)
    pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(),
        Ridge())
    param_grid = {'polynomialfeatures__degree': [1, 2, 3],
                  'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters: {}".format(grid.best_params_))
    print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))

"""
    chose better model
"""
if False:
    """
    When we wanted to skip a step in the pipeline, we can set that step to None
    """
    pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

    param_grid = [
        {
            'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
            'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        {
            'classifier': [RandomForestClassifier(n_estimators=100)],
            'preprocessing': [None], 'classifier__max_features': [1, 2, 3]
        }
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best params:\n{}\n".format(grid.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))
