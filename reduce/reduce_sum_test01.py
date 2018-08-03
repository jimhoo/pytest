'''
Created on 2018年7月4日

@author: Administrator
'''
import tensorflow as tf
import numpy as np

x=tf.constant([1,2,3,4,5],dtype=tf.float32)
y=tf.constant([5,4,3,2,1],dtype=tf.float32)

# X=tf.Variable(tf.placeholder(tf.float32))
# Y=tf.Variable(tf.placeholder(tf.float32))
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    r_x=sess.run(x)
    print('r_x:',r_x)

sess.close()