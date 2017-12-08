#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/8/13 15:19
@annotation = ''
"""
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

"""
Principal Component Analysis (PCA) 
selecting only a subset of the new features how important they are for explaining the data

PCA transfor‐ mation as rotating the data and then dropping the components with low variance

# 逆转PCA
return to the original feature space can be done using the inverse_transform method
"""

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape


print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))

from sklearn.neighbors import KNeighborsClassifier
# split the data into training and test sets
X_people = people.data[mask]
y_people = people.target[mask]
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
# build a KNeighborsClassifier using one neighbor
knn = KNeighborsClassifier(n_neighbors=1) knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))