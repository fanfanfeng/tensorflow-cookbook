# create by fanfan on 2017/7/8 0008
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

with tf.Session() as sess:
    x_vals = np.random.normal(1,0.1,100)
    y_vals = np.repeat(10.,100)

    x_data = tf.placeholder(shape=[1],dtype=tf.float32)
    y_target = tf.placeholder(shape=[1],dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[1]))
    my_output = tf.multiply(x_data,A)

    loss = tf.square(my_output - y_target)

    my_opt = tf.train.GradientDescentOptimizer(0.02)
    train_step = my_opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    loss_stochastic = []
    for i in range(100):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        feed_dict = {
            x_data:rand_x,
            y_target:rand_y
        }
        sess.run(train_step,feed_dict)
        if (i+1) % 5 == 0 :
            print("Step # " + str(i+1) + " A = " + str(sess.run(A)))
            temp_loss = sess.run(loss,feed_dict)
            print("Loss = " + str(temp_loss))
            loss_stochastic.append(temp_loss)

ops.reset_default_graph()
with tf.Session() as sess:
    batch_size = 25
    v_vals = np.random.normal(1,0.1,100)
    y_vals = np.repeat(10.,100)
    x_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
    y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[1,1]))
    my_output = tf.matmul(x_data,A)

    loss = tf.reduce_mean(tf.square(my_output - y_target))

    init = tf.global_variables_initializer()
    sess.run(init)

    my_opt = tf.train.GradientDescentOptimizer(0.02)
    train_step = my_opt.minimize(loss)

    loss_batch = []
    for i in range(100):
        rand_index = np.random.choice(100,size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        feed_dict = {
            x_data : rand_x,
            y_target: rand_y
        }
        sess.run(train_step,feed_dict)
        if(i+1)%5 == 0:
            print("Step #" + str(i+1) + " A = " + str(sess.run(A)))
            temp_loss = sess.run(loss,feed_dict)
            print("Loss = " + str(temp_loss))
            loss_batch.append(temp_loss)

plt.plot(range(0,100,5),loss_stochastic,'b-',label="Stochastic Loss")
plt.plot(range(0,100,5),loss_batch,'r--',label="Batch Loss,size=25")
plt.legend(loc="upper right",prop={'size':11})
plt.show()

