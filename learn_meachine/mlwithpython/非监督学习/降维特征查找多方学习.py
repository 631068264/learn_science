#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/8/13 15:19
@annotation = ''
"""

"""
non-negative matrix factori‐ zation (NMF), which is commonly used for feature extraction
t-SNE, which is commonly used for visualization using two-dimensional scatter plots
"""

"""
Principal Component Analysis (PCA) 
selecting only a subset of the new features how important they are for explaining the data
"""
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
