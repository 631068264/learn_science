#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/7/4 14:34
@annotation = ''
"""

"""
Naive Bayes classifiers are quite similar to the linear models


faster in training, provide generalization performance that 
is slightly worse than that of linear classifiers like LogisticRegression and LinearSVC
The models work very well with high-dimensional sparse data and are relatively robust to the parameters


There are three kinds of naive Bayes classifiers implemented in scikit-learn: 
GaussianNB, BernoulliNB, and MultinomialNB. 

GaussianNB can be applied to any continuous data, 
while BernoulliNB assumes binary data and 
MultinomialNB assumes count data (that is, that each feature represents an integer count of something, like how often a word appears in a sentence).
BernoulliNB and MultinomialNB are mostly used in text data classification.

"""
