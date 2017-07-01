# create by fanfan on 2017/7/1 0001
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

with tf.Session() as sess:
    #create data to feed in the placeholder
    x_vals = np.array([1.,3.,5.,7.,9.])

    #create the tensorflow placeholder
    x_data = tf.placeholder(tf.float32)

    # constant for multilication
    m = tf.constant(3.)

    #mutiplication
    prod = tf.multiply(x_data,m)
    for x_val in x_vals:
        print(sess.run(prod,feed_dict={x_data:x_val}))

    merged = tf.summary.merge_all(key='sumaries')
    if not os.path.exists('tensorboard_logs/'):
        os.makedirs('tensorboard_logs/')
    my_writer = tf.summary.FileWriter('tensorboard_logs/',sess.graph)
