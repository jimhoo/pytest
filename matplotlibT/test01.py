'''
Created on 2018年7月1日

@author: Administrator
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy 
if __name__ == '__main__':
#     x=np.arange(0,10,0.1);
#     y=np.sin(x);
#     plt.plot(x,y);
#     plt.show();
    x=tf.random_normal([1,5],seed=1);
    init=tf.global_variables_initializer();
    with tf.Session() as sess:
        sess.run(init)
        rst=sess.run(x);
    print("rst:",rst);
    plt.plot(rst,'bo');
    plt.show()
    sess.close()
    pass