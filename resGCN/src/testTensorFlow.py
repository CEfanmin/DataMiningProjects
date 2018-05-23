import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

x =2
y =3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y )
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)

a = tf.constant([2, 2], name='a')   # constant not fit for big data, but variable
b = tf.constant([[0,1], [2,3]], name='b')
c = tf.multiply(a, b ,name='mul')

tf.zeros([2, 3], tf.int16)
tf.fill([2, 3], 8)

d = tf.Variable(2, name='scalar')
e = tf.Variable([[0,1], [2,3]], name='matrix')
W = tf.Variable(tf.random_uniform([784, 10]), name='weight')
assign_op = d.assign(100)

# writer = tf.summary.FileWriter('../graphs', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    z, not_useless,outc = sess.run([pow_op, useless, c])
    print(sess.run(d.assign_add(2)))
    print(sess.run(d.assign_sub(2)))
    print(sess.run(assign_op))
    print(W.eval())
    print(z, not_useless)
    print(type(outc))

# writer.close()