#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/11/14 10:01
@annotation = ''
"""
# import tensorflow as tf
#
# # Save to file
# # remember to define the same dtype and shape when restore
# W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')
#
# # init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# # 替换成下面的写法:
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "my_net/save_net.ckpt")
#     print("Save to path: ", save_path)


# import numpy as np
# import tensorflow as tf
#
# # 先建立 W, b 的容器
# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     # 提取变量
#     saver.restore(sess, "my_net/save_net.ckpt")
#     print("weights:", sess.run(W))
#     print("biases:", sess.run(b))