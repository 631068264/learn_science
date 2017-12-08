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
import mglearn
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cancer.data)

# keep the first two principal components of the data
pca = PCA(n_components=2)
# fit PCA model to breast cancer data
pca.fit(X_scaled)
# transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))
print("PCA component shape: {}".format(pca.components_.shape))
# Each row in components_ corresponds to one principal component, and they are sorted by their importance
print("PCA components:\n{}".format(pca.components_))

plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()
