# create by fanfan on 2017/7/1 0001
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

with tf.Session() as sess:
    x_vals = tf.linspace(-1.,1.,500)

    # create our target of zero
    target = tf.constant(0.)

    #L2 loss
    #L = (pred - actual)^2
    l2_y_vals = tf.square(target - x_vals)
    l2_y_out = sess.run(l2_y_vals)

    #L1 loss
    #L = abs(pred - actual)
    l1_y_vals = tf.abs(target - x_vals)
    l1_y_out = sess.run(l1_y_vals)

    #pseudo loss
    #L = delta^2 * (sqrt(1+((pred - actual)/delta)^2) -1)

    deltal = tf.constant(0.25)
    phuber1_y_vals = tf.multiply(tf.square(deltal),tf.sqrt(1. + tf.square((target - x_vals)/deltal)) - 1.)
    phuber1_y_out = sess.run(phuber1_y_vals)

    deltal2 = tf.constant(0.5)
    phuber2_y_vals = tf.multiply(tf.square(deltal2), tf.sqrt(1. + tf.square((target - x_vals) / deltal2)) - 1.)
    phuber2_y_out = sess.run(phuber2_y_vals)

    x_array = sess.run(x_vals)
    plt.plot(x_array,l2_y_out,'b-',label='L2 loss')
    plt.plot(x_array,l1_y_out,'r--',label='L1 loss')
    plt.plot(x_array,phuber1_y_out,'k-.',label="P-Huber Loss(0.25)")
    plt.plot(x_array,phuber2_y_out,'g:',label="P-Huber Loss(5.0)")

    plt.ylim(-0.2,0.4)
    plt.legend(loc="lower right",prop={'size':11})
    plt.show()