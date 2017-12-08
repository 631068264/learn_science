#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/8/26 12:31
@annotation = ''
"""

"""
It tries to find cluster centers that are representative of certain regions of the data.

    assigning each data point to the closest cluster center
    setting each cluster center as the mean of the data points that are assigned to it
"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons

# # generate synthetic two-dimensional data
# X, y = make_blobs(random_state=1)
# # build the clustering model
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# print("Cluster memberships:\n{}".format(kmeans.labels_))
# print(kmeans.predict(X))
# print(kmeans.cluster_centers_)
"""
The cluster centers are stored in the cluster_centers_ attribute
"""

"""
会有不擅长的地方
performs poorly if the clusters have more complex shapes
k-means also assumes that all directions are equally important for each cluster
"""
# X_train, X_test, y_train, y_test = train_test_split(
#         X_people, y_people, stratify=y_people, random_state=0)
#
# nmf = NMF(n_components=100, random_state=0)
# nmf.fit(X_train)
# pca = PCA(n_components=100, random_state=0)
# pca.fit(X_train)
# kmeans = KMeans(n_clusters=100, random_state=0)
# kmeans.fit(X_train)
# X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
# X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
# X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)


X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)
print(y_pred)
distance_features = kmeans.transform(X)
print("Distance feature shape: {}".format(distance_features.shape))
print("Distance features:\n{}".format(distance_features))

# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60,
#             marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# print("Cluster memberships:\n{}".format(y_pred))
# plt.show()

"""
because it runs relatively quickly. k-means scales easily to large datasets, 
and scikit-learn even includes a more scalable variant in the MiniBatchKMeans class, which can handle very large datasets
"""
"""
缺点
    relies on a random initialization
    relatively restrictive assumptions made on the shape of clusters, and the requirement to specify the num‐ ber of clusters you are looking for
"""