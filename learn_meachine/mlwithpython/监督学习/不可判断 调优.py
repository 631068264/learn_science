#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/7/16 21:07
@annotation = ''
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=42)
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

print("Decision function shape: {}".format(
    gbrt.decision_function(X_test).shape))  # plot the first few entries of the decision function
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))

"""
Nearest neighbors
    For small datasets, good as a baseline, easy to explain.
Linear models
    Go-to as a first algorithm to try, 
    good for very large datasets, 
    good for very high- dimensional data.
Naive Bayes
    Only for classification. 
    Even faster than linear models, 
    good for very large data‐ sets and high-dimensional data.
    Often less accurate than linear models.
Decision trees
    Very fast, 
    don’t need scaling of the data, 
    can be visualized and easily explained.
Random forests
    Nearly always perform better than a single decision tree, very robust and powerful. 
    Don’t need scaling of data.
    Not good for very high-dimensional sparse data.
Gradient boosted decision trees
    Often slightly more accurate than random forests. 
    Slower to train but faster to predict than random forests, and smaller in memory. 
    Need more parameter tuning than random forests.
Support vector machines
    Powerful for mediumsized datasets of features with similar meaning. 
    Require scaling of data, sensitive to parameters.
Neural networks
    Can build very complex models, particularly for large datasets. 
    Sensitive to scaling of the data and to the choice of parameters. 
    Large models need a long time to train.
"""
