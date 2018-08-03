'''
Created on 2018年6月12日

@author: Administrator
'''
import numpy as np
import tensorflow as tf
from numpy.core.multiarray import dtype

if __name__ == '__main__':

    input=tf.Variable( tf.ones([1,2,2,2]));
    filter=tf.Variable(tf.ones([1,1,2,1]));
    op=tf.nn.conv2d(input, filter, [1,1,1,1], padding='VALID');
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables());
        print(sess.run(input))
        print(sess.run(filter))
        rst1=sess.run(op);
        print(rst1);
    sess.close();
    pass