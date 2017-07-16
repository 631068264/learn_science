#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/7/3 17:15
@annotation = ''
"""
import mglearn
from sklearn.model_selection import train_test_split

# iris = load_iris()
# x = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, random_state=0)
# print x.shape
# print x.feature_names
# print("X_train shape: {}".format(X_train.shape))
# print("y_train shape: {}".format(y_train.shape))
# print("y_train shape: {}".format(X_test.shape))
# print("y_train shape: {}".format(y_test.shape))
# X, y = mglearn.datasets.make_forge()
# # plot dataset
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(["Class 0", "Class 1"], loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# print("X.shape: {}".format(X.shape))
# plt.show()

"""KNN决策边界"""
# X, y = mglearn.datasets.make_forge()
# # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# for n_neighbors, ax in zip([1, 3, 9], axes):
#     # the fit method returns the object self, so we can instantiate
#     # and fit in one line
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{} neighbor(s)".format(n_neighbors))
#     ax.set_xlabel("feature 0")
#     ax.set_ylabel("feature 1")
# axes[0].legend(loc=3)

"""KNN学习曲线"""
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=66)
# training_accuracy = []
# test_accuracy = []
# # try n_neighbors from 1 to 10
# neighbors_settings = range(1, 11)
# for n_neighbors in neighbors_settings:
#     # build the model
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train, y_train)
#     # record training set accuracy
#     training_accuracy.append(clf.score(X_train, y_train))
#     # record generalization accuracy
#     test_accuracy.append(clf.score(X_test, y_test))
#
# plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
# plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("n_neighbors")
# plt.legend()
# plt.show()