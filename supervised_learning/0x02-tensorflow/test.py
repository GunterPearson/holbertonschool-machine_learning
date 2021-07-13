#!/usr/bin/env python3
""" test """
import tensorflow as tf

string = tf.Variable("this is a string", tf.string) 
rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok", "hello"], ["test", "yes", "hello"], ["test", "yes", "hello"]], tf.string)

tensor1 = tf.ones([1,2,3]) # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])
# x = tf.rank(tensor1)
with tf.Session() as sess:
    print(tensor1.eval())
    print(tensor1)
    # print(sess.run(x))
    print(tensor2.eval())
    print(tensor2)
    print(tensor3.eval()[:1])
    print(tensor3)

