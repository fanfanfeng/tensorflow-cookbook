# create by fanfan on 2017/7/6 0006
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32,shape=[None,1],name="x")
y = tf.placeholder(tf.float32,shape=[None,1],name='y')
w = tf.Variable(tf.constant(0.0))

global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(0.1,global_step,10,2,staircase=False)
loss = tf.pow(w*x -y,2)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        sess.run(train_step,feed_dict={x:np.linspace(1,2,10).reshape([10,1]),y:np.linspace(1,2,10).reshape([10,1])})
        print(sess.run(learning_rate))
        print(sess.run(global_step))