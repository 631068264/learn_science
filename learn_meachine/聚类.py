#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/5 16:41
@annotation = ''
"""
import numpy as np
import scipy as sp
from sklearn import cluster, datasets, decomposition
from sklearn.feature_extraction.image import grid_to_graph

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

if False:
    # 大概分成3类
    k_means = cluster.KMeans(n_clusters=3)
    k_means.fit(X_iris)
    print(k_means.labels_[::10])

    print(y_iris[::10])

if False:

    try:
        face = sp.face(gray=True)
    except AttributeError:
        from scipy import misc

        face = misc.face(gray=True)

    X = face.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    k_means = cluster.KMeans(n_clusters=5, n_init=1)
    k_means.fit(X)

    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    print(values)
    print(labels)
    print(X.shape)
    face_compressed = np.choose(labels, values)
    face_compressed.shape = face.shape
    print(face_compressed)

if False:
    """Hierarchical agglomerative clustering"""
    # 建立聚类的层次结构

    # 集聚 - 自下而上的方法：每个观察在自己的集群中开始，集群被迭代地合并，
    # 以最小化连接标准。当感兴趣的集团仅由少数观察结果组成时，
    # 这种方法特别有意义。当簇的数量较大时，其计算效率要高于k - means

    from sklearn.utils.testing import SkipTest
    from sklearn.utils.fixes import sp_version

    if sp_version < (0, 12):
        raise SkipTest("Skipping because SciPy version earlier than 0.12.0 and "
                       "thus does not include the scipy.misc.face() image.")

    ###############################################################################
    # Generate data
    try:
        face = sp.face(gray=True)
    except AttributeError:
        # Newer versions of scipy have face in misc
        from scipy import misc

        face = misc.face(gray=True)

    # Resize it to 10% of the original size to speed up the processing
    face = sp.misc.imresize(face, 0.10) / 255.

if False:
    """特征聚合 减少维度灾难"""
    digits = datasets.load_digits()
    images = digits.images
    X = np.reshape(images, (len(images), -1))
    print(X.shape)
    connectivity = grid_to_graph(*images[0].shape)
    agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
                                         n_clusters=32)
    agglo.fit(X)
    # reduce the dimensionality of the dataset
    X_reduced = agglo.transform(X)
    print(X_reduced.shape)

    X_approx = agglo.inverse_transform(X_reduced)
    print(X_approx.shape)
    images_approx = np.reshape(X_approx, images.shape)

if False:
    """成分分析"""

    """
    PCA Principal component analysis selects the successive components
    that explain the maximum variance in the signal.

    Independent component analysis (ICA
    独立组件分析（ICA）选择组件，使其负载的分布具有最大量的独立信息。它能够恢复 非高斯独立信号：
    """
    # Create a signal with only 2 useful dimensions
    x1 = np.random.normal(size=100)
    x2 = np.random.normal(size=100)
    x3 = x1 + x2
    X = np.c_[x1, x2, x3]

    pca = decomposition.PCA()
    pca.fit(X)
    # PCA查找数据不平坦的方向
    # 用于变换数据时，PCA可以通过在主子空间上投影来降低数据的维数
    print(pca.explained_variance_)

    # As we can see, only the 2 first components are useful
    pca.n_components = 2
    X_reduced = pca.fit_transform(X)
    print(X_reduced.shape)

    time = np.linspace(0, 10, 2000)
    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    S = np.c_[s1, s2]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise
    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    # Compute ICA
    ica = decomposition.FastICA()
    S_ = ica.fit_transform(X)  # Get the estimated sources
    A_ = ica.mixing_.T
    print(np.allclose(X, np.dot(S_, A_) + ica.mean_))
