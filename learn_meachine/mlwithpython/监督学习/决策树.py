#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/7/4 14:58
@annotation = ''
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

"""
Decision Trees they learn a hierarchy of if/else questions, leading to a decision


There are two common strategies to prevent overfitting: 
    1.stopping the creation of the tree early (also called pre-pruning)
    
    limit the depth tree/leaves number/min 分支
    
    2.building the tree but then removing or collapsing nodes that contain little information 
    (also called post-pruning or just pruning)
    
Method 参数:max_depth, max_leaf_nodes, or min_samples_leaf—is sufficient to prevent overfitting. 
This leads to a lower accuracy on the training set, but an improvement on the test set
"""


# from sklearn.tree import DecisionTreeClassifier
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# tree = DecisionTreeClassifier(max_depth=4, random_state=0)
# # tree = DecisionTreeClassifier( random_state=0)
# tree.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
# print("Feature importances:\n{}".format(tree.feature_importances_))
#
#
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


#
#
# plot_feature_importances_cancer(tree)


"""
Decision trees is that they tend to overfit the training data
Random forests are one way to address this problem

随机深林
A random forest is essentially a collection of decision trees, 
where each tree is slightly different from the others

you need to decide on the number of trees to build 
(the n_estimators parameter of RandomForestRegressor or RandomForestClassifier)

we would use many more trees (often hundreds or thousands), leading to even smoother boundaries



Random forests don’t tend to perform well on very high dimensional, sparse data, such as text data.
random forests require more memory and are slower to train and to predict than linear models

max_features determines how random each tree is, and a smaller max_features reduces overfitting.

max_features=sqrt(n_features) for classification and max_fea tures=log2(n_features) for regression. 
Adding max_features or max_leaf_nodes might sometimes improve performance
"""
from sklearn.datasets import load_breast_cancer

# X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
#                                                     random_state=42)
# forest = RandomForestClassifier(n_estimators=5, random_state=2)
# forest.fit(X_train, y_train)
#
# fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#     ax.set_title("Tree {}".format(i))
#     mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
#
# mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
#                                 alpha=.4)
# axes[-1, -1].set_title("Random Forest")
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.show()

# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, random_state=0)
# forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
# forest.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
# plot_feature_importances_cancer(forest)

"""
Gradient boosted regression trees

By default, there is no randomization in gradient boosted regression trees; 
instead, strong pre-pruning is used. 
Gradient boosted trees often use very shallow trees, of depth one to five, 
which makes the model smaller in terms of memory and makes predictions faster.

another impor‐ tant parameter of gradient boosting is the learning_rate, 
which controls how strongly each tree tries to correct the mistakes of the previous trees

To reduce overfitting, we could either apply stronger pre-pruning by limiting the maximum depth or lower the learning rate

先试验随机深林 predict慢或者精确率不高 使用梯度上升

使用GradientBoostingClassifier处理大数据 考虑使用xgboost faster than scikit-learn
"""
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
plot_feature_importances_cancer(gbrt)

"""
a long time to train
does not work well on high-dimensional sparse data

递归上升
The main parameters of gradient boosted tree models are the number of trees, 
n_estimators, and the learning_rate, which controls the degree to which each tree is allowed to 
correct the mistakes of the previous trees.
a lower learning_rate means that more trees are needed to build a model of similar complexity

随机深林
higher n_esti mators value is always better, 
increasing n_estimators in gradient boosting leads to a more complex model, which may lead to overfitting.

参数定下n_estimators & learning_rates & max_depth
A common practice is to fit n_estimators depending on the time and memory budget, 
and then search over different learning_rates.
max_depth (or alternatively max_leaf_nodes), to reduce the complexity of each tree. 
Usually max_depth is set very low for gradient boosted models, often not deeper than five splits

"""
