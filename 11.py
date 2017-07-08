# create by fanfan on 2017/7/5 0005
import tensorflow as tf

with tf.Session() as sess:
    initializer = tf.random_uniform_initializer(5, 20)
    with tf.variable_scope("test", initializer=initializer):
        aa = tf.get_variable("aa", shape=[2, 2], dtype=tf.int32)
    sess.run(tf.global_variables_initializer())
    print(sess.run(aa))