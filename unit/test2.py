'''
Created on 2018年6月12日

@author: Administrator
'''
import numpy as np
import tensorflow as tf

if __name__ == '__main__':

    a=tf.Variable(tf.random_normal([3,5],seed=1))
    b=tf.Variable(tf.truncated_normal([2,2], seed=2))
    # dtype, shape, name, verify_shape
    c=tf.constant([1,2,3],shape=[2,3])
    init=tf.global_variables_initializer()
#     sess=tf.Session();

    with tf.Session() as sess:
        sess.run(init)
        rst1=sess.run(a);
        rst2=sess.run(b);
        rst3=sess.run(c);
        print(rst1)
        print(rst2)
        print(rst3)
    sess.close()
    pass