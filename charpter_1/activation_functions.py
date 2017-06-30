# create by fanfan on 2017/6/30 0030
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
with tf.Session() as sess:
    #x range
    x_vals = np.linspace(start=-10.,stop=10.,num=100)

    #relu activation
    print(sess.run(tf.nn.relu([-3,3,10])))
    y_relu = sess.run(tf.nn.relu(x_vals))

    #relu-6 activation
    print(sess.run(tf.nn.relu6([-3,3,10])))
    y_relu6 = sess.run(tf.nn.relu6(x_vals))

    # sigmoid activation
    print(sess.run(tf.nn.sigmoid([-1.,0.,1.])))
    y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

    #Hyper Tangent activation
    print(sess.run(tf.nn.tanh([-1.0,0,1])))
    y_tanh = sess.run(tf.nn.tanh(x_vals))

    # Softsign activation
    print(sess.run(tf.nn.softsign([-1.,0.,1.])))
    y_softsign = sess.run(tf.nn.softsign(x_vals))

    #Softplus activation
    print(sess.run(tf.nn.softplus([-1.,0.,1.])))
    y_softplus = sess.run(tf.nn.softplus(x_vals))

    # Exponential linear activation
    print(sess.run(tf.nn.elu([-1.,0.,1.])))
    y_elu = sess.run(tf.nn.elu(x_vals))

    plt.plot(x_vals,y_softplus,"r--",label='Softplus',linewidth=2)
    plt.plot(x_vals,y_relu,'b:',label="Relu",linewidth=2)
    plt.plot(x_vals,y_relu6,"g-",label="RELU6",linewidth=2)
    plt.plot(x_vals,y_elu,'k-',label="ExpLU",linewidth=0.5)
    plt.ylim([-1.5,7])
    plt.legend(loc='top left')
    plt.show()


    plt.plot(x_vals,y_sigmoid,'r--',label='Sigmoid',linewidth=2)
    plt.plot(x_vals,y_tanh,"b:",label='Tanh',linewidth=2)
    plt.plot(x_vals,y_softsign,'g-.',label='Softsign',linewidth=2)
    plt.ylim([-2,2])
    plt.legend(loc='top left')
    plt.show()