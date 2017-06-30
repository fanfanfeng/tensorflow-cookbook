# create by fanfan on 2017/6/30 0030
import tensorflow as  tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

with tf.Session() as sess:
    print("div() as truediv() as floordiv()")
    print(sess.run(tf.div(3,4)))
    print(sess.run(tf.truediv(3,4)))
    print(sess.run(tf.floordiv(3.0,4.0)))

    print("\n Mode function")
    print(sess.run(tf.mod(22.0,5.0)))

    print("\nCrooss Product")
    print(sess.run(tf.cross([1.,0.,0.],[0.,1.,0.])))

    print("\nTrig functions")
    print(sess.run(tf.sin(3.1416)))
    print(sess.run(tf.cos(3.1416)))

    print("\nTangemt")
    print(sess.run(tf.div(tf.sin(3.1416/4.),tf.cos(3.1416/4.))))

    print("\nCustom operation")
    test_nums = range(15)
    def custom_polynomial(x_val):
        # Return 3x~2 -x + 10
        return(tf.subtract(3* tf.square(x_val),x_val) + 10)

    print(sess.run(custom_polynomial(11)))

    expected_output = [3*x*x - x + 10 for x in test_nums]
    print(expected_output)
    for num in test_nums:
        print(sess.run(custom_polynomial(num)))