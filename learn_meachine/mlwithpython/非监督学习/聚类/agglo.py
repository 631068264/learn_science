#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/8/26 22:16
@annotation = ''
"""
import mglearn
from sklearn.datasets import make_blobs

"""
Agglomerative clustering

each point is its own cluster
merges the two most similar clusters until is the number of clusters
"""
"""
How to specify the “most similar cluster”

ward
    The default choice, ward picks the two clusters to merge such that 
    the variance within all clusters increases the least. This often leads to clusters that are relatively equally sized.
average
    average linkage merges the two clusters that have the smallest average distance between all their points.
complete
    complete linkage (also known as maximum linkage) merges the two clusters that have the smallest maximum distance between their points.

ward works on most datasets
if one is much bigger than all the others, for example), average or complete might work better

"""

"""
Because of the way the algorithm works, 
agglomerative clustering cannot make predictions for new data points
Therefore, Agglomerative Clustering has no predict method
use the fit_predict method instead to  build the model and get the cluster member‐ ships on the training set
"""
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

X, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3)
# fix and return labels_
assignment = agg.fit_predict(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
