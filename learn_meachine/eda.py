#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/17 22:29
@annotation = ''
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

plt.style.use('ggplot')
iris = datasets.load_iris()

X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())

# pd.scatter_matrix(df,  figsize=[8, 8], s=150, marker='D')
pd.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='D')

plt.show()
