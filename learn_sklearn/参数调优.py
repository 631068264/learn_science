#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/18 12:33
@annotation = ''
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

"""
Hyperparameter tuning
● Linear regression: Choosing parameters
● Ridge/lasso regression: Choosing alpha
● k-Nearest Neighbors: Choosing n_neighbors
● Parameters like alpha and k: Hyperparameters
● Hyperparameters cannot be learned by fi!ing the model

Grid search cross-validation
"""

digits = datasets.load_digits()
X = digits.data
y = digits.target

param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)

print knn_cv.best_params_
print knn_cv.best_score_

"""
Hold-out set reasoning
● How well can the model perform on never before seen data?
● Using ALL data for cross-validation is not ideal
● Split data into training and hold-out set at the beginning
● Perform grid search cross-validation on training set
● Choose best hyperparameters and evaluate on hold-out set
"""
