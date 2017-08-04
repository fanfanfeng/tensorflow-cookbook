# create by fanfan on 2017/7/27 0027
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import  ops
ops.reset_default_graph()


with tf.Session() as sess:
    iris = datasets.load_iris()
    x_vals = np.array([x[3] for x in iris.data])
    y_vals = np.array([y[0] for y in iris.data])

    batch_size = 125
    x_data = tf.placeholder(shape=[None,1],dtype=tf.float32)
    y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    model_output = tf.add(tf.matmul(x_data,A),b)

    demming_numerator = tf.abs(tf.subtract(tf.add(tf.matmul(x_data,A),b),y_target))
    demming_denominator = tf.sqrt(tf.add(tf.square(A),1))
    loss = tf.reduce_mean(tf.truediv(demming_numerator,demming_denominator))

    my_opt = tf.train.GradientDescentOptimizer(0.25)
    train_step = my_opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec = []
    for i in range(500):
        rand_index = np.random.choice(len(x_vals),size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
        temp_loss = sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
        loss_vec.append(temp_loss)
        if (i+1)%100 == 0:
            print("Step #" + str(i+1) + " A = " + str(sess.run(A)) + " b = " + str(sess.run(b)))
            print("Loss = " + str(temp_loss)
                  )

    [slope] = sess.run(A)
    [y_intercept] = sess.run(b)

    best_fit = []
    for i in x_vals:
        best_fit.append(slope* i + y_intercept)


    plt.plot(x_vals,y_vals,'o',label="Data Points")
    plt.plot(x_vals,best_fit,'r-',label="Best fit line",linewidth=3)

    plt.legend(loc="upper left")
    plt.title('Sepal Length vas Pedal width')
    plt.xlabel('Pedal width')
    plt.ylabel('sepal length')
    plt.show()

    plt.plot(loss_vec,'k-')
    plt.title('Demming Loss per generation')
    plt.xlabel('Iteration')
    plt.ylabel('Demming Loss')
    plt.show()