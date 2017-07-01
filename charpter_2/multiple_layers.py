# create by fanfan on 2017/7/1 0001
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops
ops.reset_default_graph()

# define a custon layer which will be sigmoid(Ax +b) where
# x is a 2x2 matrix and A and b are 2x2 matrices
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1.,2.],[-1,3.]])
    b = tf.constant(1.,shape=[2,2])
    temp1 = tf.matmul(A,input_matrix_sqeezed)
    temp = tf.add(temp1,b) # Ax + b
    return (tf.sigmoid(temp))

with tf.Session() as sess:

    x_shape = [1,4,4,1]
    x_val = np.random.uniform(size=x_shape)

    x_data = tf.placeholder(tf.float32,shape=x_shape)

    my_filter = tf.constant(0.25,shape=[2,2,1,1])
    my_strides = [1,2,2,1]
    mov_avg_layer = tf.nn.conv2d(x_data,my_filter,my_strides,padding='SAME',name='Moving_Avg_Window')

    #add custom layer to graph
    with tf.name_scope('Custom_layer') as scope:
        custom_layer1 = custom_layer(mov_avg_layer)

    print(sess.run(mov_avg_layer,feed_dict={x_data:x_val}))

    print(sess.run(custom_layer1,feed_dict={x_data:x_val}))

    merged = tf.summary.merge_all(key="summaries")
    if not os.path.exists('tensorboard_logs/'):
        os.makedirs('tensorboard_logs/')

    my_writer = tf.summary.FileWriter('tensorboard_logs',sess.graph)