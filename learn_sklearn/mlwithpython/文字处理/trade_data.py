#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/11 16:38
@annotation = ''
"""
import numpy as np
from sklearn.datasets import load_files

reviews_train = load_files("data/train/")
# load_files returns a bunch, containing training texts and training labels
text_train, y_train = reviews_train.data, reviews_train.target
print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[1]:\n{}".format(text_train[1]))
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
print("Samples per class (training): {}".format(np.bincount(y_train)))

"""
分词
标号
统计词频
"""
