# create by fanfan on 2017/7/25 0025
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

with tf.Session() as sess:
    x_vals = np.linspace(0,10,100)
    y_vals = x_vals + np.random.normal(0,1,100)

    x_vals_column = np.transpose(np.matrix(x_vals))
    ones_column = np.transpose(np.matrix(np.repeat(1,100)))
    A = np.column_stack((x_vals_column,ones_column))

    y = np.transpose(np.matrix(y_vals))

    A_tensor = tf.constant(A)
    y_tensor = tf.constant(y)

    tA_A  = tf.matmul(tf.transpose(A_tensor),A_tensor)
    L = tf.cholesky(tA_A)

    tA_y = tf.matmul(tf.transpose(A_tensor),y)
    sol1 = tf.matrix_solve(L,tA_y)

    sol2 = tf.matrix_solve(tf.transpose(L),sol1)

    solution_eval = sess.run(sol2)

    slope = solution_eval[0][0]
    y_intercept = solution_eval[1][0]

    print("slope: " + str(slope))
    print('y_intercepte: ' + str(y_intercept))

    best_fit = []
    for i in x_vals:
        best_fit.append(slope *i + y_intercept)

    plt.plot(x_vals,y_vals,'o',label='Data')
    plt.plot(x_vals,best_fit,'r-',label='Best fit line',linewidth=3)
    plt.legend(loc='upper left')
    plt.show()

