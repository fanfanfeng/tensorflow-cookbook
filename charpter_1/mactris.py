# create by fanfan on 2017/6/29 0029
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

with tf.Session() as sess:
    identity_matrix = tf.diag([1.0,1.0,1.0])
    print(sess.run(identity_matrix))

    #2 X3 random norm matrix
    A = tf.truncated_normal([2,3])
    print(sess.run(A))

    #2x3 constant matrix
    B = tf.fill([2,3],5.0)
    print(sess.run(B))

    #3x2 random uniform matrix
    C = tf.random_uniform([3,2])
    print(sess.run(C))
    #diff with before
    print(sess.run(C))

    #create matrix from np array
    D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
    print(sess.run(D))

    #Matrix addition/subtraction
    print(sess.run(A + B))
    print(sess.run(B - B))

    #matrix Multiplication
    print(sess.run(tf.matmul(B,identity_matrix)))

    #matrix Transpose
    print(sess.run(tf.transpose(C)))


    #matrix Determinant
    print(sess.run(tf.matrix_determinant(D)))

    #matrix inverse
    print(sess.run(tf.matrix_inverse(D)))

    #Cholesky decomposition
    print(sess.run(tf.cholesky(identity_matrix)))

    #Eigenvalues and Eigenvectors
    #特征值和特征向量
    print(sess.run(tf.self_adjoint_eig(D)))