# create by fanfan on 2017/6/29 0029
import tensorflow as tf

row_dim = 4
col_dim = 4
#Create a zero filled tensor
zero_tsr = tf.zeros([row_dim,col_dim])

#Create a one filled tensor.
ones_tsr = tf.ones([row_dim,col_dim])

#Create a constant filled tensor.
filled_tsr = tf.fill([row_dim,col_dim],42)

#Create a tensor out of an existing constant.
constant_tsr = tf.constant([1,2,3])

##Tensors of similar shape:
#新建一个与给定的tensor类型大小一致的tensor，其所有元素为0。
zeros_similar = tf.zeros_like(constant_tsr)

#返回一个tensor，该tensor中的数值在start到stop区间之间取等差数列（包含start和stop），如果num>1则差值为(stop-start)/(num-1)，以保证最后一个元素的值为stop。
linear_tsr = tf.linspace(start=0.0, stop=1.0,num=3) #result  [0.0, 0.5, 1.0]

# 返回一个tensor等差数列，该tensor中的数值在start到limit之间，不包括limit，delta是等差数列的差值。
integer_seq_ter = tf.range(start=6,limit=15,delta=3) #result [6, 9, 12].

#返回一个形状为shape的tensor，其中的元素服从minval和maxval之间的均匀分布。
#( minval <=  x <  maxval ).
randunif_tsr = tf.random_uniform([row_dim,col_dim],minval=0,maxval=1)

#返回一个形状为shapede tensor,其中元素服从 均值为0，方差为1的正态分布。
randnorm_tsr = tf.random_normal([row_dim,col_dim],mean=0.0,stddev=1.0)

#返回一个形状为shapede tensor,其中元素服从 均值为0，方差为1的正态分布。
# x在区间（μ-2σ，μ+2σ）之间
runcnorm_tsr = tf.truncated_normal([row_dim, col_dim],mean=0.0, stddev=1.0)

#对value（是一个tensor）的第一维进行随机化
shuffled_output = tf.random_shuffle(runcnorm_tsr)
#随机剪裁输入的tensor到指定size
cropped_output = tf.random_crop(runcnorm_tsr,size=[2,2])


with tf.Session() as sess:
    first_var = tf.Variable(tf.zeros([2,3]))
    print(sess.run(first_var.initializer))

    second_var = tf.Variable(tf.zeros_like(first_var))
    print(sess.run(second_var.initializer))



