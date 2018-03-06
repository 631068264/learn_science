#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/5 10:40
@annotation = ''
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

"""
how good each feature is? 
There are three basic strategies: 
    univariate statistics,
    model-based selection, 
    iterative selection
特征值多次出现，有明确含义，排除不该存在的值
They need the target for fitting the model. 
This means we need to split the data into training and test sets, and fit the feature selection only on the training part of the data
"""

"""
statistically significant relation‐ ship between each feature and the target
Consequently, a feature will be discarded if it is only informative when combined with another feature
Univariate tests are often very fast to compute, and don’t require building a model.
"""

cancer = load_breast_cancer()
# get deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
print noise[:5]
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=.5)
"""
with the simplest ones being SelectKB est, which selects a fixed number k of features
SelectPercentile, which selects a fixed percentage of features
"""
# use f_classif (the default) and SelectPercentile to select 50% of features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# transform training set
X_train_selected = select.transform(X_train)

print("cancer.data.shape: {}".format(cancer.data.shape))
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))
# We can find out which features have been selected using the get_support method, which returns a Boolean mask of the selected features
"""
In this case, removing the noise features improved performance, even though some of the original features were lost
"""
mask = select.get_support()
print(mask)
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
# plt.show()

"""
Model-Based Feature Selection
needs to provide some measure of importance for each feature,
"""
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

# 、The SelectFromModel class selects all features that
# have an importance measure of the feature (as provided by the supervised model) greater than the provided threshold
select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="median")
"""
Iterative Feature Selection

while in model-based selection we used a single model to select features. 
In iterative feature selection, a series of models are built, with varying numbers of features

两种情况
starting with no features and adding features one by one until some stopping criterion is reached 
starting with all features and removing features one by one until some stopping criterion is reached 
    【recursive feature elimination (RFE)】

"""
from sklearn.feature_selection import RFE

select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
             n_features_to_select=40)
select.fit(X_train, y_train)
# visualize the selected features:
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.show()
