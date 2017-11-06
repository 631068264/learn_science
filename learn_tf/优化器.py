#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/11/4 21:41
@annotation = ''
"""
from __future__ import print_function

import tensorflow as tf


# kx+b
def create_loss(k, b):
    pass


k = .3
b = -.3

# Model parameters
W = tf.Variable([k], dtype=tf.float32)
b = tf.Variable([b], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares

raw_loss = loss
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(raw_loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, raw_loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

fixW = tf.assign(W, curr_W)
fixb = tf.assign(b, curr_b)
sess.run([fixW, fixb])
print(sess.run(loss, {x: x_train, y: y_train}))
