#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/11/13 14:34
@annotation = ''
"""
from __future__ import print_function

import tensorflow as tf

matrix1 = tf.constant([
    [3, 3]
])
matrix2 = tf.constant([
    [2],
    [2]
])

product = tf.matmul(matrix1, matrix2)
"""
会话控制
"""
if False:
    sess = tf.Session()
    result = sess.run(product)
    print(result)

if False:
    with tf.Session() as sess:
        result = sess.run(product)
        print(result)

"""
变量
"""
if False:
    state = tf.Variable(0, name='counter')

    # 定义常量 one
    one = tf.constant(1)

    # 定义加法步骤 (注: 此步并没有直接计算)
    new_value = tf.add(state, one)

    # 将 State 更新成 new_value
    update = tf.assign(state, new_value)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))

if False:
    """
    placeholder
    """
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={
            input1: 7,
            input2: 2,
        }))

if True:
    pass