#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/8/27 10:28
@annotation = ''
"""
from sklearn.datasets import make_blobs

"""
DBSCAN (which stands for “density- based spatial clustering of applications with noise
does not require the user to set the number of clusters
somewhat slower than agglomerative clustering and k-means

DBSCAN works by identifying points where many data points are close together
"""

"""
There are two parameters in DBSCAN: min_samples and eps

If there are at least min_samples many data points within a distance of eps to a given data point, 
that data point is classified as a core sample
Core samples that are closer to each other than the distance eps are put into the same cluster by DBSCAN 
"""

"""
In the end, there are three kinds of points: core points, points that are within distance eps of core points (called boundary points), and noise.

assigned the label -1, which stands for noise
"""

from sklearn.cluster import DBSCAN

X, y = make_blobs(random_state=0, n_samples=12)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("Cluster memberships:\n{}".format(clusters))
"""
Increasing eps means that more points will be included in a cluster

Increasing min_samples means that fewer points will be core points, 
and more points will be labeled as noise

for eps is sometimes easier after scaling the data using StandardScaler or MinMaxScaler
"""
