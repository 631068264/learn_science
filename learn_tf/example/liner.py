#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/11/13 14:15
@annotation = ''
"""
from __future__ import print_function

import numpy as np
import tensorflow as tf

X = np.random.rand(100).astype(np.float32)
y = 0.1 * X + 3

# init structure
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

t_y = W * X + biases
loss = tf.reduce_mean(tf.square(y - t_y))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# end

sess = tf.Session()
sess.run(init)

for step in range(301):
    sess.run(train)
    if step % 10 == 0:
        print(sess.run(W), sess.run(biases))
