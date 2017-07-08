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


    x_vals = tf.linspace(-3.,5.,500)
    target = tf.constant(1.)
    targets = tf.fill([500,],1.)

    # Hing loss
    # L = max(0,1 - (pred * actual))
    hinge_y_vals = tf.maximum(0.,1. - tf.multiply(target,x_vals))
    hinge_y_out = sess.run(hinge_y_vals)

    # Cross entropy loss
    # L = -actual*(log(pred)) - (1-actual)(log(1 -pred))
    xentropy_y_vals = - tf.multiply(target,tf.log(x_vals)) - tf.multiply((1. - target),tf.log(1. - x_vals))
    xentropy_y_out = sess.run(xentropy_y_vals)

    # sigmoid Entroy loss
    # L = -actual * (log(sigmode(pred)) - (1 - actual)(log(1 -sigmoid(pred))
    x_val_input = tf.expand_dims(x_vals,1)
    target_input = tf.expand_dims(targets,1)
    xentropy_sigmoid_y_vals = tf.nn.softmax_cross_entropy_with_logits(logits=x_val_input,labels=target_input)
    xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

    # Weighted softmax cross entropy loss
    # L = -actual * (log(pred)) * weights - (1-actual)(log(1-pred))
    weight = tf.constant(0.5)
    xentropy_weightd_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals,targets,weight)
    xentropy_weightd_y_vals = sess.run(xentropy_weightd_y_vals)

    x_array = sess.run(x_vals)
    plt.plot(x_array,hinge_y_out,'b-',label="Hinge loss")
    plt.plot(x_array,xentropy_y_out,'r--',label="Cross Entropy Loss")
    plt.plot(x_array,xentropy_sigmoid_y_out,'k-.',label="Cross Entropy sigmoid")
    plt.plot(x_array,xentropy_sigmoid_y_out,'g:',label="Weighted cross entropy sigmoid")
    plt.ylim(-1.5,3)
    plt.legend(loc='lower right',prop={'size':11})
    plt.show()