#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/7/29 21:43
@annotation = ''
"""
"""
StandardScaler 
    in scikit-learn ensures that for each feature the mean is 0 and the variance is 1, 
    bringing all features to the same magnitude
    
    this scaling does not ensure any particular minimum and maximum values for the features
    
MinMaxScaler
    this means all of the data is contained within the rectangle created by the x-axis between 0 and 1 
    and the y-axis between 0 and 1
    
Normalizer
    In other words, it projects a data point on the circle (or sphere, in the case of higher dimensions) 
    with a radius of 1
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
