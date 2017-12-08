#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/8/14 22:10
@annotation = ''
"""

"""
Non-Negative Matrix Factorization (NMF)

extract useful features & dimensionality reduction

In PCA write each data point as a weighted sum of some components

PCA NMF不同
in PCA we wanted components that were orthogonal and that explained as much variance of the data as possible, 
in NMF, we want the components and the coefficients to be non-negative; 
        we want both the components and the coefficients >= 0
能够区分混合数据
NMF can identify the original components that make up the combined data
all components play an equal part

NMF uses a random initialization, which might lead to different results depending on the random seed
components is lower than the number of input features


t-SNE
from sklearn.manifold import TSNE
可以更好绘图
manifold learning algorithms that allow for much more complex mappings, and often provide better visualizations
"""
