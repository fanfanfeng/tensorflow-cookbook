# create by fanfan on 2017/7/8 0008
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

with tf.Session() as sess:
    batch_size = 25
    x_vals = np.random.normal(1,0.1,100)
    y_vals = np.repeat(10.,100)
    x_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
    y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)

    train_indices = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]

    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    A = tf.Variable(tf.random_normal(shape=[1,1]))
    my_output = tf.matmul(x_data,A)


    loss = tf.reduce_mean(tf.square(my_output - y_target))
    my_opt = tf.train.GradientDescentOptimizer(0.02)
    train_step = my_opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(100):
        rand_index = np.random.choice(len(x_vals_train),size=batch_size)
        rand_x = np.transpose([x_vals_train[rand_index]])
        rand_y = np.transpose([y_vals_train[rand_index]])
        feed_dict = {
            x_data:rand_x,
            y_target:rand_y
        }

        sess.run(train_step,feed_dict)
        if (i+1) %25 == 0:
            print("Step # " + str(i+1) + " A = " + str(sess.run(A)))
            print("Loss = " + str(sess.run(loss,feed_dict)))


    mse_test = sess.run(loss,feed_dict={x_data:np.transpose([x_vals_test]),y_target:np.transpose([y_vals_test])})
    mse_train = sess.run(loss,feed_dict={x_data:np.transpose([x_vals_train]),y_target:np.transpose([y_vals_train])})
    print(" MSE  on test : " + str(np.round(mse_test,2)))
    print(" MSe on train : " + str(np.round(mse_train,2)))


ops.reset_default_graph()
with tf.Session() as sess:
    batch_size = 25
    x_vals = np.concatenate((np.random.normal(-1,1,50),np.random.normal(1,1,50)))
    y_vals  = np.concatenate((np.repeat(0,50),np.repeat(1.,50)))
    x_data  = tf.placeholder(shape=[1,None],dtype=tf.float32)
    y_target = tf.placeholder(shape=[1,None],dtype=tf.float32)

    train_indices = np.random.choice(len(x_vals),round(len(x_vals) * 0.8),replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vars_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    A = tf.Variable(tf.random_normal(mean=10,shape=[1]))

    my_output = tf.add(x_data,A)

    xencropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output,labels=y_target))

    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(xencropy)

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(3000):
        rand_index = np.random.choice(len(x_vals_train),size=batch_size)
        rand_x = [x_vals_train[rand_index]]
        rand_y = [y_vals_train[rand_index]]
        feed_dict = {
            x_data:rand_x,
            y_target:rand_y
        }
        sess.run(train_step,feed_dict)

        if (i+1)%200 == 0:
            print("Step # " + str(i+1) + " A = " + str(sess.run(A)))
            print("Loss = " + str(sess.run(xencropy,feed_dict)))


    y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data,A))))
    correct_prediction  = tf.equal(y_prediction,y_target)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    acc_value_test = sess.run(accuracy,feed_dict={x_data:[x_vals_test],y_target:[y_vals_test]})
    acc_value_train = sess.run(accuracy,feed_dict={x_data:[x_vals_train],y_target:[y_vals_train]})

    print("Accuracy on train set : " + str(acc_value_train))
    print("Accuracy on test set: " + str(acc_value_test))



